"""
    angularFOF.py
    lsskit.catio

    __author__ : Nick Hand
    __email__  : nhand@berkeley.edu
    __desc__   : module to compute angular clustering using a friends-of-friends algorithm
"""
from . import kdtree, numpy as np
from catIO import catalog

import pickle
import os
import collections
import copy
import contextlib

#-------------------------------------------------------------------------------
clusteringResult = collections.namedtuple('clusteringResult', ['groups',  
                                                               'radius',
                                                               'tag'])


defaultCols = {'group_number' : {'desc' : 'the group number', 'order' : 0, 'type' : int, 'fmt' : '%10d', 'default': -999},\
               'x0'           : {'desc' : 'central x coordinate of the group', 'order' : 1, 'type' : float, 'fmt' : '%15.5f', 'default': -999.}, \
               'y0'           : {'desc' : 'central y coordinate of the group', 'order' : 2, 'type' : float, 'fmt' : '%15.5f', 'default' : -999.}, \
               'size'         : {'desc' : 'number of galaxies in group', 'order' : 3, 'type' : int, 'fmt' : '%10d', 'default': -999}}

#-------------------------------------------------------------------------------
class Group(object):
    """ 
    A group object is a collection of sources
    """

    def __init__(self, group_number):
        """
        Parameters
        ----------
        group_number : int
            the number of the group used to identify it 

        """
        self.size         = 0            # number of galaxies
        self.members      = []           # list of members in group
        self.radius       = 0            # radius in Mpc
        self.x0, self.y0  = 0., 0.       # center cartesian coordinates
        self.group_number = group_number # the group number 
    #end __init__
    
    #--------------------------------------------------------------------------- 
    def findMember(self, member):
        """
        Return the member galaxy if found, or None otherwise
        
        Parameters
        ----------
        member : source.source 
            the member to search for
        """
        try: 
            index = self.members.index(member)
        except:
            return None
            
        return self.members[index]
    #end findMember
    
    #---------------------------------------------------------------------------    
    def add (self, g):
        """
        Add a galaxy to the group.
        
        Parameters
        ----------
        g : source.source 
            galaxy object to add to self
        """     
        # make a new copy
        g = copy.deepcopy(g)
           
        # keep track of group_number
        g.host = self.group_number
        
        # add member to list
        self.members.append(g)

        # increment size of galaxies
        self.size += 1

        # find new centroid
        self.x0 = (self.x0*(self.size - 1) + g.x) / self.size
        self.y0 = (self.y0*(self.size - 1) + g.y) / self.size
                        
        rad = np.sqrt( (self.x0 - g.x)**2 + (self.y0 - g.y)**2 )
        if rad > self.radius: 
            self.radius = rad

    #end add
    
    #---------------------------------------------------------------------------              
    def addGroup (self, gr):
        """ 
        Add a group to self
        
        Parameters
        ----------
        gr : Group 
            the group object to add
        
        Returns
        -------
        merged : bool
            returns True if group was added successfully
        """ 
        # make a new copy
        gr = copy.deepcopy(gr)
        
        # calculate new centroid
        self.x0 = (self.size*self.x0 + gr.size*gr.x0)/(self.size+gr.size)
        self.y0 = (self.size*self.y0 + gr.size*gr.y0)/(self.size+gr.size)
        
        # combine size of members
        self.size = self.size + gr.size

        # combine member lists
        self.members += gr.members
        
        # update radius
        self.radius = 0.
        for i in range(len(self.members)):
            self.members[i].host = self.group_number
            mem = self.members[i]
            rad = np.sqrt( (self.x0 - mem.x)**2 + (self.y0 - mem.y)**2 )
            if rad > self.radius:
                self.radius = rad
    
        return True
    #end addGroup

    #---------------------------------------------------------------------------
    def center (self):
        """ 
        Returns the group center in cartesian coordinates (x0, y0)
        """
        return self.x0, self.y0
    #end centerDeg
    
    #---------------------------------------------------------------------------
    def __str__ (self):
        """
        The string representation of the group
        """       
        return "%d: %.3f %.3f" %(self.group_number, self.x0, self.y0)
    #end __str__
#endclass Group

#-------------------------------------------------------------------------------
class groupList( dict ):
    """
    A class that acts as a list of Groups for helping to facilitate I/O actions
    """

    def save( self, filename ):
        """
        Save the groupList as a pickle to filename
        """
        f = file(filename, 'w')
        pickle.dump( self, f )
        f.close()
    #end save

    #---------------------------------------------------------------------------
    def read( self, filename ):
        """
        Read a groupList from a pickle file
        """
        f = file(filename)
        gr = pickle.load( f )
        self += gr
        f.close()
    #end read
        
    #---------------------------------------------------------------------------
    def isEmpty(self):
        """
        Return True if self is empty
        """
        return len(self) == 0
    #end isEmpty
    
    #---------------------------------------------------------------------------
    def append(self, val):
        
        if not hasattr(val, 'group_number'):
            raise AttributeError("Group to append must have `group_number` attribute.")

        if self.has_key(val.group_number):
            raise ValueError("Group number `%d` already exists in this groupList" %val.group_number)
            
        # add the group with key = group_number
        self[val.group_number] = val
    #end append
    
    #---------------------------------------------------------------------------
#endclass groupList
    
#-------------------------------------------------------------------------------        
class groupFinder(object):
    """
    Find the groups in a catalog object and create a group catalog using the 
    friends-of-friends algorithm
    """

    def __init__(self, coord_keys=['x', 'y'], nprocs=1):

        self.groups = groupList()
        self.coord_keys = coord_keys
        self.nprocs = nprocs
                            
    #end __init__
        
    #---------------------------------------------------------------------------
    def addGalaxies(self, galaxy_df):
        """
        Add the input list of galaxies to groups list ``self.groups``
        
        Parameters
        ----------
        galaxy_list : pandas.DataFrame
            pandas DataFrame containing information for galaxies to run FOF
            algorithm on
        """
        print("adding galaxies...")
        #bar = utilities.initializeProgressBar(len(galaxy_df))
        
        # add the 'host' column
        subsample = galaxy_df[self.coord_keys]
        subsample['host'] = np.zeros(len(subsample))
        
        # now make it into a recarray
        info = subsample.to_records(index=True)
        
        # now make the groups
        for cnt, gal in enumerate(info):
        
            #bar.update(cnt+1)
            
            # make a new group and add to end of self.groups
            newgr = Group(len(self.groups)) 
                   
            # update host group numbers and then add to the new group
            gal.host = newgr.group_number
            newgr.add(gal)
        
            # add the new group to the group list
            self.groups.append(newgr)
            
    #end addGalaxies
        
    #---------------------------------------------------------------------------
    def mergeGroupsRecursive(self, this_group, neighbor_list, tree):
        """
        Recursively merge together lists of neighboring source
        """
        if len(neighbor_list) == 0:
            return     
        
        for neighbor_index in neighbor_list:
            
            # this is the neighbor source
            neighbor = tree.objects[neighbor_index]
            original_neighbor_host = neighbor.host
            
            # merge this group with the neighbor's group
            if this_group.group_number != original_neighbor_host:
                
                self.groups[this_group.group_number].addGroup(self.groups[original_neighbor_host]) 
                
                # now remove the merged group from self.groups
                self.groups.pop(original_neighbor_host)
                
                # update this objects host group number
                tree.objects[neighbor_index].host = this_group.group_number
                
                # if we merged, redundant groups exist
                self.redundantGroupsExist = True

                # now merge the neighbor list of the neighbor to the same group
                self.mergeGroupsRecursive(self.groups[this_group.group_number], self.neighbor_lists[neighbor_index], tree)
            
    #---------------------------------------------------------------------------
    def findGroups(self, radius):
        """ 
        Find groups using the friends-of-friends algorithm
        """
        # save the radius for later
        self.radius = radius
            
        # initialize the kd-tree for nearest neighbor searches
        sources = [g for gr_num in self.groups for g in self.groups[gr_num].members]
                
        # make the coordinate attributes exist
        if not all(hasattr(sources[0], col) for col in self.coord_keys):
            raise ValueError("Error accessing coordinate columns %s" %self.coord_keys)

        # initialize the parallel kdtree
        tree = kdtree.KDTreeSources(sources, fields=self.coord_keys, angular=False, nprocs=self.nprocs)
        print("kd-tree size = %d" %tree.size)
            
        # get ALL the neighbors, using cartesian radius
        print("finding all neighbors...")
        self.neighbor_lists = tree.range(sources, radius, radius_type='cartesian')
                
        # now loop over each list of neighbors
        self.redundantGroupsExist = True
        passes = 1
        while self.redundantGroupsExist:
            
            self.redundantGroupsExist = False
            print("merging groups, pass #%d..." %passes)

            # initialize the progress bar
            #bar = utilities.initializeProgressBar(tree.size)
            
            # recursively merge groups
            for source_num, neighbor_list in enumerate(self.neighbor_lists):
                
                #bar.update(source_num+1)
                
                # do the recursive merging here
                this_group_num = tree.objects[source_num].host
                self.mergeGroupsRecursive(self.groups[this_group_num], neighbor_list, tree)
                
            passes += 1
            
        n_groups = 0
        for gr_num, gr in self.groups.items(): 
            if gr.size > 1: n_groups += 1
            
        print("number of groups with size > 1 = ", n_groups)
        return self.groups
    #end findGroups
    
    #---------------------------------------------------------------------------
    def saveGroups(self, tag, catCols=defaultCols, min_size=1):
        """
        Save the groups that have been made
        """
        with contextlib.suppress(OSError):
            os.makedirs('catalogs')
                                                        
        # remove groups with size < min_size
        if min_size > 1:
            for gr_num in self.groups:
                if self.groups[gr_num].size < min_size:
                    self.groups.pop(gr_num)
        
            
        fname = "catalogs/groups_%s_perp_%.4f.pickle" %(tag, self.radius)
        results = clusteringResult(self.groups, self.radius, tag)
        pickle.dump(results, open(fname, 'w'))
        
        # initialize an output catalog
        outCat = catalog.catalog(cols=catCols)

        # write out the catalog
        for gr_num in self.groups:
            
            gr = self.groups[gr_num]
            outCat.addRow({'group_number': gr_num, 'x0': gr.x0, 'y0': gr.y0, 'size': gr.size})

        outCat.sortBy('size')
        outCat.reverse()

        fname = "catalogs/groups_top_level_%s_perp_%.4f.pickle" %(tag, self.radius)
        outCat.write(fname)
#endclass groupFinder

#-------------------------------------------------------------------------------
