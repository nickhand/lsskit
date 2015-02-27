"""
 kdtree.py
 lss : code to implement a kd-tree, in parallel too!
 
 author: Nick Hand
 contact: nhand@berkeley.edu
 creation date: 02/26/2015
"""

import ctypes
import multiprocessing as mp
import copy
import numpy as np
import scipy.spatial

if hasattr(scipy.spatial, 'cKDTree') and hasattr(scipy.spatial.cKDTree, 'query_ball_tree'):
    tree = scipy.spatial.cKDTree
else:
    print "Warning: scipy.spatial.cKDTree outdated; operations will be slower"
    tree = scipy.spatial.KDTree

#-------------------------------------------------------------------------------
def shmem_as_ndarray(raw_array):
    """
    View shared memory region as ``numpy.ndarray`` object
    """
    _ctypes_to_numpy = {
        ctypes.c_char : np.int8,
        ctypes.c_wchar : np.int16, 
        ctypes.c_byte : np.int8, 
        ctypes.c_ubyte : np.uint8, 
        ctypes.c_short : np.int16, 
        ctypes.c_ushort : np.uint16, 
        ctypes.c_int : np.int32, 
        ctypes.c_uint : np.int32, 
        ctypes.c_long : np.int32, 
        ctypes.c_ulong : np.int32, 
        ctypes.c_float : np.float32, 
        ctypes.c_double : np.float64 
    }
    address = raw_array._wrapper.get_address()
    size = raw_array._wrapper.get_size()
    dtype = _ctypes_to_numpy[raw_array._type_]

    class Dummy(object): pass
    d = Dummy()
    d.__array_interface__ = {
        'data' : (address, False),
        'typestr' : np.dtype(np.uint8).str,
        'descr' : np.dtype(np.uint8).descr,
        'shape' : (size, ),
        'strides' : None,
        'version' : 3
    }
    return np.asarray(d).view(dtype=dtype)
#end shmem_as_ndarray

#-------------------------------------------------------------------------------
class Scheduler(object):
    
    def __init__(self, ndata, nprocs, chunk=None, schedule='guided'):
        if not schedule in ['guided', 'dynamic', 'static']:
            raise ValueError, 'unknown scheduling strategy'
        
        self._ndata = mp.RawValue(ctypes.c_int, ndata)
        self._start = mp.RawValue(ctypes.c_int, 0)
        self._lock = mp.Lock()
        self._schedule = schedule
        self._nprocs = nprocs
        if schedule == 'guided' or schedule == 'dynamic':
            min_chunk = ndata // (10*nprocs)
            if chunk:
                min_chunk = chunk
            min_chunk = 1 if min_chunk < 1 else min_chunk
            self._chunk = min_chunk
        elif schedule == 'static':
            min_chunk = ndata // nprocs
            if chunk: 
                min_chunk = chunk if chunk > min_chunk else min_chunk
            min_chunk = 1 if min_chunk < 1 else min_chunk
            self._chunk = min_chunk
    
    def __iter__(self):
        return self
        
    def next(self):
        
        with self._lock:
            ndata = self._ndata.value
            nprocs = self._nprocs
            start = self._start.value
            if self._schedule == 'guided':
                _chunk = ndata // nprocs
                chunk = max(self._chunk, _chunk)
            else:
                chunk = self._chunk
        
            if ndata:
                if chunk > ndata:
                    s0 = start
                    s1 = start + ndata
                    self._ndata.value = 0
                else:
                    s0 = start
                    s1 = start + chunk
                    self._ndata.value = ndata - chunk
                    self._start.value = start + chunk
                return slice(s0, s1)
            else:
                raise StopIteration
#endclass Scheduler

#-------------------------------------------------------------------------------
def _parallel_query(scheduler, data, ndata, ndim, leafsize, x, nx, d, i, 
                    k, eps, p, dub, ierr):
    """
    This is the worker function that does the parallel query
    """
    try: 
        # view shared memory as ndarrays. 
        _data = shmem_as_ndarray(data).reshape((ndata,ndim)) 
        _x = shmem_as_ndarray(x).reshape((nx,ndim)) 
        _d = shmem_as_ndarray(d).reshape((nx,k)) 
        _i = shmem_as_ndarray(i).reshape((nx,k)) 

        # reconstruct the kd-tree from the data. 
        # this is relatively inexpensive. 
        kdtree = tree(_data, leafsize=leafsize) 

        # query for nearest neighbours, using slice ranges, 
        # from the load balancer. 

        for s in scheduler: 
            a, b = kdtree.query(_x[s,:], k=k, eps=eps, p=p, distance_upper_bound=dub) 
            if k == 1:
                a = a.reshape(len(a), 1)
                b = b.reshape(len(b), 1)
            _d[s,:], _i[s,:] = a, b
    # an error occured, increment the return value ierr. 
    # access to ierr is serialized by multiprocessing. 
    except: 
        ierr.value += 1 
#end _parallel_query

#-------------------------------------------------------------------------------
def _parallel_query_ball_tree(scheduler, data, ndata, ndim, leafsize, x, nx, d, 
                                r, eps, p, ierr):
    """
    This is the worker function that does the parallel query_ball_tree
    """
    try: 
        # view shared memory as ndarrays. 
        _data = shmem_as_ndarray(data).reshape((ndata,ndim)) 
        _x = shmem_as_ndarray(x).reshape((nx,ndim)) 

        # reconstruct the kd-tree from the data. 
        # this is relatively inexpensive. 
        kdtree = tree(_data, leafsize=leafsize) 
    
        # query for nearest neighbours, using slice ranges, 
        # from the load balancer. 
        for s in scheduler: 
        
            # and add the other points
            other = tree(_x[s,:], leafsize=leafsize)
            d[s] = other.query_ball_tree(kdtree, r, p=p, eps=eps)
    
    # an error occured, increment the return value ierr. 
    # access to ierr is serialized by multiprocessing. 
    except: 
        ierr.value += 1 
#end _parallel_query_ball_tree
     
#-------------------------------------------------------------------------------
class KDTree_MP(tree):
    """
    A parallel wrapper class around scipy.spatial.cKDTree
    """
    def __init__(self, data, leafsize=10, nprocs=1):
        """
        Notes
        -----
        Same as cKDTree.__init__ except that an internal copy of data to 
        shared memory is made
        
        Parameters
        ----------
        data : array-like, shape (n,m)
            The n data points of dimension m to be indexed.
        leafsize : positive integer
            The number of points at which the algorithm switches over to
            brute-force.
        nprocs : int
            The number of processors to use. Default is 1.
        """
        self.nprocs = nprocs
        
        n, m = data.shape
        
        # allocate shared memory for data 
        self.shmem_data = mp.RawArray(ctypes.c_double, n*m)
        
        # view shared memory as ndarray, and copy over the data. 
        # The RawArray objects have information about the dtype and 
        # buffer size. 
        _data = shmem_as_ndarray(self.shmem_data).reshape((n,m)) 
        _data[:,:] = data
        
        
        # initialize parent, we must do this last because 
        # cKDTree stores a reference to the data array. We pass in 
        # the copy in shared memory rather than the origial data. 
        self._leafsize = leafsize 
        super(KDTree_MP, self).__init__(_data, leafsize=leafsize) 
    
        # initialize self.objects to None
        self.objects = None
    #end __init__
    
    #---------------------------------------------------------------------------
    def _parallel_query(self, x, k=1, eps=0, p=2, distance_upper_bound=np.inf, 
                        chunk=None, schedule='guided'):
        """
        Same as cKDTree.query except parallelized with multiple 
        processes and shared memory. 

        Notes
        -----
        See ``self.nearest`` documentation for parameter specifications.
        """
        
        # allocate shared memory for x and result 
        nx = x.shape[0] 
        shmem_x = mp.RawArray(ctypes.c_double, nx*self.m) 
        shmem_d = mp.RawArray(ctypes.c_double, nx*k) 
        shmem_i = mp.RawArray(ctypes.c_int, nx*k) 
        
        # view shared memory as ndarrays 
        _x = shmem_as_ndarray(shmem_x).reshape((nx,self.m)) 
        _d = shmem_as_ndarray(shmem_d).reshape((nx,k)) 
        _i = shmem_as_ndarray(shmem_i).reshape((nx,k))
        
        # copy x to shared memory 
        _x[:] = x
        
        # set up a scheduler to load balance the query 
        scheduler = Scheduler(nx, self.nprocs, chunk=chunk, schedule=schedule) 
        
        # return status in shared memory 
        # access to these values are serialized automatically 
        ierr = mp.Value(ctypes.c_int, 0) 
        err_msg = mp.Array(ctypes.c_char, 1024)
        
        
        # query with multiple processes 
        query_args = (scheduler, 
                        self.shmem_data, self.n, self.m, self.leafsize, 
                        shmem_x, nx, shmem_d, shmem_i, 
                        k, eps, p, distance_upper_bound, 
                        ierr) 
        query_fun = _parallel_query 
        
        # make the pool of processors and start/join them
        pool = [mp.Process(target=query_fun, args=query_args) for n in range(self.nprocs)] 
        for p in pool: p.start() 
        for p in pool: p.join() 
        
        if ierr.value != 0: 
            raise RuntimeError, ('%d errors in worker processes. Last one reported:\n%s' 
                                    % (ierr.value, err_msg.value)) 

        # return results (private memory) 
        if k == 1:
            _d = _d.reshape((_d.shape[0],))
            _i = _i.reshape((_i.shape[0],))
        return _d.copy(), _i.copy()
    #end parallel_query
    
    #---------------------------------------------------------------------------
    def _parallel_query_ball_tree(self, x, r, p=2., eps=0., chunk=None, schedule='guided'):
        """
        Similiar to cKDTree.query_ball_tree except parallelized with multiple 
        processes and shared memory. 
        
        Notes
        -----
        See ``self.range`` documentation for parameter specifications.
        """
        
        # allocate shared memory for x and result 
        nx = x.shape[0] 
        shmem_x = mp.RawArray(ctypes.c_double, nx*self.m) 
        shmem_d = mp.Manager().list(xrange(nx))
        
        # view shared memory as ndarrays 
        _x = shmem_as_ndarray(shmem_x).reshape((nx, self.m)) 
        
        # copy x to shared memory 
        _x[:] = x
        
        # set up a scheduler to load balance the query 
        scheduler = Scheduler(nx, self.nprocs, chunk=chunk, schedule=schedule) 
        
        # return status in shared memory 
        # access to these values are serialized automatically 
        ierr = mp.Value(ctypes.c_int, 0) 
        err_msg = mp.Array(ctypes.c_char, 1024)
        
        
        # query with multiple processes 
        query_args = (scheduler, 
                        self.shmem_data, self.n, self.m, self.leafsize, 
                        shmem_x, nx, shmem_d, 
                        r, eps, p, ierr)
                         
        query_fun = _parallel_query_ball_tree
        
        # make the pool of processors and start/join them
        pool = [mp.Process(target=query_fun, args=query_args) for n in range(self.nprocs)] 
        for p in pool: p.start() 
        for p in pool: p.join() 
        
        if ierr.value != 0: 
            raise RuntimeError, ('%d errors in worker processes. Last one reported:\n%s' 
                                    % (ierr.value, err_msg.value)) 

        # return results (private memory)     
        return copy.copy(list(shmem_d))
    #end parallel_query_ball_tree
        
    #---------------------------------------------------------------------------
    @property
    def size(self):
        """
        The size of the kd-tree
        """
        return len(self.data)
    #end size        
    #---------------------------------------------------------------------------
    def peek(self, obj):
        """
        Return the latest version of the input object or None if the tree 
        does not contain the object
        """
        if self.objects is not None:
            d = np.array(self.objects)
        else:
            d = self.data
            
        # test whether all array elements along axis 1 evaluate to True
        inds = np.where((d == obj).all(axis=1)) 
        out = d[inds]
        if len(out) == 0:
            return None
        elif len(out) > 1:
            raise ValueError("Duplicate objects detected in peek()")
        else:
            return out
    #end peek
    
    #---------------------------------------------------------------------------
    def range(self, pts, r, p=2., eps=0., chunk=None, schedule='guided'):
        """
        Find all pairs of points whose distance is at most ``r``. This will call
        ``cKDTree.query_ball_tree``.
        
        Parameters
        ----------
        pts : array_like, last dimension self.m
            For each point in the the array, search against ``self`` to find
            all neighbors in ``self.data``
        r : positive float
            The radius of points to return.
        p : float, optional
            Which Minkowski norm to use.  `p` has to meet the condition
            ``1 <= p <= infinity``.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Extra keyword arguments: 

        chunk : Minimum chunk size for the load balancer. 
        schedule: Strategy for balancing work load 
        ('static', 'dynamic' or 'guided'). 
            
        Returns
        -------
        results : list of lists
            For each element pts[i], results[i] is a list of the indices of 
            its neighbors in ``self.data``.
        """
        N, D = pts.shape
        assert(D == self.m)
        
        return self._parallel_query_ball_tree(pts, r, p=p, eps=eps, chunk=chunk, schedule=schedule)
    #end range
    
    #---------------------------------------------------------------------------
    def nearest(self, pts, k=1, eps=0, p=2, distance_upper_bound=np.inf, 
                chunk=None, schedule='guided'):
        """
        Query the kd-tree for nearest neighbors
        
        Parameters
        ----------
        pts : array_like, last dimension self.m
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the kth returned value 
            is guaranteed to be no further than (1+eps) times the 
            distance to the real k-th nearest neighbor.
        p : float, 1<=p<=infinity
            Which Minkowski p-norm to use. 
            1 is the sum-of-absolute-values "Manhattan" distance
            2 is the usual Euclidean distance
            infinity is the maximum-coordinate-difference distance
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Extra keyword arguments: 

        chunk : Minimum chunk size for the load balancer. 
        schedule: Strategy for balancing work load 
        ('static', 'dynamic' or 'guided'). 
        
        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors. 
            If x has shape tuple+(self.m,), then d has shape tuple+(k,).
            Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The locations of the neighbors in self.data.
            If `x` has shape tuple+(self.m,), then `i` has shape tuple+(k,).
            Missing neighbors are indicated with self.n.
        """
        return self._parallel_query(pts, k=k, eps=eps, p=p, 
                                    distance_upper_bound=distance_upper_bound, 
                                    chunk=chunk, schedule=schedule)
    #end nearest
    
    #---------------------------------------------------------------------------
    def update_info(self, indices, functionToRun):
        """
        Update information stored in ``self.objects``
        
        Parameters
        ----------
        indices : list, or list of lists
            list of indices of the objects to update in ``self.objects``
        functionToRun : callable
            A function to run on the matched objects that reads in the objects
            index in ``self.data`` and applies changes to ``self.objects``
        """
        if self.objects is None:
            pass
        else:
            indices = np.asarray(indices)
            if not any(isinstance(el, list) for el in indices):           
                for index in indices:
                    try:
                        functionToRun(index, self.objects)
                    except:
                        pass
            else:
                for inds in indices:
                    for index in inds:
                        try:
                            functionToRun(index, self.objects)
                        except:
                            pass
    #end update_info
    
#-------------------------------------------------------------------------------
#endclass KDTree_MP

#-------------------------------------------------------------------------------
class KDTreeSources(KDTree_MP):
    """
    A class to implement a kdtree to use with objects representing 
    astronomical sources and using Euclidean distances
    """
    
    def __init__(self, sources, fields=None, angular=True, **kwargs):
        """
        Parameters
        ----------
        sources : list
           list of objects, with attributes specifying the relevant information
           about the location of each object
        fields : list, optional
            list of the attributes holding location information
        angular : bool, optional
           If ``True``, the input attributes specify angular coordinates (in degrees), 
           otherwise the locations are assumed to be cartesian. Default is 
           ``True``.
        """        
        self.angular = angular
        self.fields = fields
        
        # convert the input to the right format
        data = self._handle_input(sources)
        
        # initialize the super class
        super(KDTreeSources, self).__init__(data, **kwargs) 
        
        # save the objects
        if isinstance(sources, (tuple, list, np.ndarray)):
            self.objects = sources
        else:
            self.objects = [sources]
    #end __init__
    
    #---------------------------------------------------------------------------
    def _handle_input(self, sources):
        """
        Internal function to format the input data
        """
        if not isinstance(sources, (tuple, list, np.ndarray)):
            sources = [sources]

        if self.fields is not None:
            coords = []
            for field in self.fields: 
                coord = np.array([getattr(g, field) for g in sources], dtype=float)
                coords.append(coord)
      
            # stack each coordinate vertical (row-wise)
            X = np.transpose(np.vstack(coords))
            
        else:

            # assume input data is np.ndarray of coordinates already
            X = sources.copy()
            
        if self.angular:

            # Convert 2D RA/DEC to 3D cartesian coordinates on the unit sphere
            X *= (np.pi/180.)  # in degrees now
            X = np.transpose(np.vstack([ np.cos(X[:, 0])*np.cos(X[:, 1]),
                                            np.sin(X[:, 0])*np.cos(X[:, 1]),
                                            np.sin(X[:, 1])]))            
        return X
    #end _handle_input
    
    #---------------------------------------------------------------------------
    def range(self, pts, r, radius_type='angular', eps=0., 
                chunk=None, schedule='guided'):
        """
        Find all pairs of points whose distance is at most ``r``. This will call
        ``cKDTree.query_ball_tree``.
        
        Parameters
        ----------
        pts : array_like, last dimension self.m
            For each point in the the array, search against ``self`` to find
            all neighbors in ``self.data``
        r : positive float
            The radius of points to return.
        radius_type : {`angular`, `cartesian`}, optional
            The type of radius specified. If `angular`, assumed to be in 
            degrees.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Extra keyword arguments: 

        chunk : Minimum chunk size for the load balancer. 
        schedule: Strategy for balancing work load 
        ('static', 'dynamic' or 'guided'). 
            
        Returns
        -------
        results : list of lists
            For each element pts[i], results[i] is a list of the indices of 
            its neighbors in ``self.data``.
        """
        if radius_type not in ['angular', 'cartesian']:
            raise ValueError("The `radius_type` parameter must be one of [`angular`, `cartesian`]")
            
        # convert the input points to the correct format
        newpts = self._handle_input(pts)
        
        # check if we need to convert the radius
        if radius_type == 'angular':
        
            if not self.angular:
                raise ValueError("Potential coordinate system differences when "
                                    "not using angular coordiantes with an angular radius.")
                                    
            # convert the angular radius to a 3D physical distance
            # assuming in degrees
            # uses law of cosines on unit sphere
            r = self.angle_to_cartesian(r)
        
        N, D = newpts.shape
        assert(D == self.m)
        
        # use cartesian distances
        return self._parallel_query_ball_tree(newpts, r, p=2, eps=eps, chunk=chunk, schedule=schedule)
    #end range
    
    #---------------------------------------------------------------------------
    def nearest(self, pts, radius_type='angular', k=1, eps=0,
                distance_upper_bound=np.inf, chunk=None, schedule='guided'):
        """
        Query the kd-tree for nearest neighbors
        
        Parameters
        ----------
        pts : array_like, last dimension self.m
            An array of points to query.
        radius_type : {`angular`, `3D`}, optional
            The type of radius specified. If `angular`, assumed to be in 
            degrees.
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the kth returned value 
            is guaranteed to be no further than (1+eps) times the 
            distance to the real k-th nearest neighbor.
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Extra keyword arguments: 

        chunk : Minimum chunk size for the load balancer. 
        schedule: Strategy for balancing work load 
        ('static', 'dynamic' or 'guided'). 
        
        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors (in degrees if ``self.angular=True``)
            If x has shape tuple+(self.m,), then d has shape tuple+(k,).
            Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The locations of the neighbors in self.data.
            If `x` has shape tuple+(self.m,), then `i` has shape tuple+(k,).
            Missing neighbors are indicated with self.n.
        """
        if radius_type not in ['angular', 'cartesian']:
            raise ValueError("The `radius_type` parameter must be one of [`angular`, `cartesian`]")
            
        # convert the input points to the correct format
        newpts = self._handle_input(pts)
        
        if not np.isinf(distance_upper_bound):
            if radius_type == 'angular':
                
                if not self.angular:
                    raise ValueError("Potential coordinate system differences when "
                                        "not using angular coordiantes with an angular radius.")
                                        
                distance_upper_bound = self.angle_to_cartesian(distance_upper_bound)
                
        # use cartesian distances
        return self._parallel_query(newpts, k=k, eps=eps, p=2, 
                                    distance_upper_bound=distance_upper_bound, 
                                    chunk=chunk, schedule=schedule)
    #end nearest
    
    #---------------------------------------------------------------------------
    def angle_to_cartesian(self, angular_dist):
        """
        Convert an angular distance (in degrees) to a 3D cartesian 
        distance, assuming the points separated by the distance are on 
        the unit sphere
        """
        return np.sqrt(2. - 2. * np.cos(angular_dist*np.pi/180.))
    #end angle_to_cartesian
    
    #---------------------------------------------------------------------------
    def cartesian_to_angle(self, cartesian_dist):
        """
        Convert a cartesian distance to an angular distance, assuming the 
        points separated by the distance are on the unit sphere
        """
        x = 0.5*cartesian_dists
        return (180. / np.pi) * 2. * np.arctan2(x, np.sqrt(np.maximum(0, 1 - x**2)))
    #end cartesian_to_angle
    
#-------------------------------------------------------------------------------
#endclass KDTreeSources

#-------------------------------------------------------------------------------    
class KDTreePixels(KDTree_MP):
    """
    A class to implement a kdtree to use with objects representing
    map pixels and using Manhattan distances
    """

    def __init__(self, pixels, fields=None, **kwargs):
        """
        Parameters
        ----------
        pixels : list
           list of objects, with attributes specifying the relevant information
           about the location of each object
        fields : list, optional
            list of the attributes holding pixel location information
        """
        self.fields = fields
        
        # convert the input to the right format
        data = self._handle_input(pixels)
        
        # initialize the super class
        super(KDTreePixels, self).__init__(data, **kwargs) 
        
        # save the objects
        self.objects = np.asarray(pixels)
    #end __init__
    
    #---------------------------------------------------------------------------
    def _handle_input(self, pixels):
        """
        Internal function to format the input data
        """
        pixels = np.array(pixels, copy=False, ndmin=1)
        if self.fields is not None:
            coords = []
            for field in self.fields:
                coord = np.array([getattr(g, field) for g in pixels], dtype=float)
                coords.append(coord)
      
            # stack each coordinate vertical (row-wise)
            X = np.transpose(np.vstack(coords))
            
        else:

            # assume input data is np.ndarray of coordinates already
            X = pixels.copy()
                        
        return X
    #end _handle_input
    
    #---------------------------------------------------------------------------
    def range(self, pts, r, eps=0., chunk=None, schedule='guided'):
        """
        Find all pairs of points whose distance is at most ``r``. This will call
        ``cKDTree.query_ball_tree``.
        
        Parameters
        ----------
        pts : array_like, last dimension self.m
            For each point in the the array, search against ``self`` to find
            all neighbors in ``self.data``
        r : positive float
            The radius of points to return.
        eps : float, optional
            Approximate search.  Branches of the tree are not explored
            if their nearest points are further than ``r/(1+eps)``, and
            branches are added in bulk if their furthest points are nearer
            than ``r * (1+eps)``.  `eps` has to be non-negative.

        Extra keyword arguments: 

        chunk : Minimum chunk size for the load balancer. 
        schedule: Strategy for balancing work load 
        ('static', 'dynamic' or 'guided'). 
            
        Returns
        -------
        results : list of lists
            For each element pts[i], results[i] is a list of the indices of 
            its neighbors in ``self.data``.
        """
        # convert the input points to the correct format
        newpts = self._handle_input(pts)
        
        N, D = newpts.shape
        assert(D == self.m)
        
        # use manhattan distances
        return self._parallel_query_ball_tree(newpts, r, p=1, eps=eps, chunk=chunk, schedule=schedule)
    #end range

    #---------------------------------------------------------------------------
    def nearest(self, pts, k=1, eps=0, distance_upper_bound=np.inf, 
                    chunk=None, schedule='guided'):
        """
        Query the kd-tree for nearest neighbors
        
        Parameters
        ----------
        pts : array_like, last dimension self.m
            An array of points to query.
        k : integer
            The number of nearest neighbors to return.
        eps : non-negative float
            Return approximate nearest neighbors; the kth returned value 
            is guaranteed to be no further than (1+eps) times the 
            distance to the real k-th nearest neighbor.
        distance_upper_bound : nonnegative float
            Return only neighbors within this distance.  This is used to prune
            tree searches, so if you are doing a series of nearest-neighbor
            queries, it may help to supply the distance to the nearest neighbor
            of the most recent point.

        Extra keyword arguments: 

        chunk : Minimum chunk size for the load balancer. 
        schedule: Strategy for balancing work load 
        ('static', 'dynamic' or 'guided'). 
        
        Returns
        -------
        d : array of floats
            The distances to the nearest neighbors (in degrees if ``self.angular=True``)
            If x has shape tuple+(self.m,), then d has shape tuple+(k,).
            Missing neighbors are indicated with infinite distances.
        i : ndarray of ints
            The locations of the neighbors in self.data.
            If `x` has shape tuple+(self.m,), then `i` has shape tuple+(k,).
            Missing neighbors are indicated with self.n.
        """
        # convert the input points to the correct format
        newpts = self._handle_input(pts)
        
        # use manhattan distances
        return self._parallel_query(newpts, k=k, eps=eps, p=1, 
                                    distance_upper_bound=distance_upper_bound, 
                                    chunk=chunk, schedule=schedule)
        #end nearest
        #-----------------------------------------------------------------------
        
#endclass KDTreePixels
#-------------------------------------------------------------------------------
    
    

