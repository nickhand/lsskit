import logging
import os
import traceback
import string
import functools

from pyRSD.rsdfit.util import rsdfit_parser
from pyRSD.rsdfit import rsdfit, params_filename
from mpi4py import MPI

from . import lib

#------------------------------------------------------------------------------
# tools
#------------------------------------------------------------------------------        
def rsetattr(obj, attr, val):
    """
    Recursive setattr for multiple layers of attributes
    """
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)
    
sentinel = object()
def rgetattr(obj, attr, default=sentinel):
    """
    Recursive getattr for multiple layers of attributes
    """
    if default is sentinel:
        _getattr = getattr
    else:
        def _getattr(obj, name):
            return getattr(obj, name, default)
    return functools.reduce(_getattr, [obj]+attr.split('.'))    
    
def split_ranks(N_ranks, N):
    """
    Divide the ranks into chunks, attempting to have `N` ranks
    in each chunk. This removes the master (0) rank, such 
    that `N_ranks - 1` ranks are available to be grouped
    
    Parameters
    ----------
    N_ranks : int
        the total number of ranks available
    N : int
        the desired number of ranks per worker
    """
    available = list(range(1, N_ranks)) # available ranks to do work
    total = len(available)
    extra_ranks = total % N
  
    for i in range(total//N):
        yield i, available[i*N:(i+1)*N]
    
    if extra_ranks and extra_ranks >= N//2:
        remove = extra_ranks % 2 # make it an even number
        ranks = available[-extra_ranks:]
        if remove: ranks = ranks[:-remove]
        if len(ranks):
            yield i+1, ranks
        
def enum(*sequential, **named):
    enums = dict(zip(sequential, range(len(sequential))), **named)
    return type('Enum', (), enums)
    
#------------------------------------------------------------------------------
# class to run rsdfit in batch mode
#------------------------------------------------------------------------------
class RSDFitBatch(object):
    """
    Task manager that will iterate over ``rsdfit`` computations,
    possibly in parallel using MPI
    """
    logger = logging.getLogger('rsdfit-batch')
    
    def __init__(self, comm, 
                       cpus_per_worker,
                       command, 
                       task_keys, 
                       task_values, 
                       log_level=logging.INFO,
                       print_output=False):
        """
        Parameters
        ----------
        comm : MPI communicator
            the global communicator that will be split and divided
            amongs the independent workers
        cpus_per_worker : int
            the desired number of ranks assigned to each independent
            worker, when iterating over the tasks in parallel
        command : RSDFitBatchCommand
            the batch ``rsdfit`` command instance
        task_keys : list
            a list of strings specifying the names of the task dimensions -- 
            these specify the string formatting key when updating the config
            template file for each task value
        task_values : list
            a list of tuples specifying the task values which will be iterated 
            over -- each tuple should be the length of `task_keys`
        log_level : int, optional
            an integer specifying the logging level to use -- default
            is the `INFO` level
        print_output : bool, optional
            just print the intended output directory and exit
        """
        self.logger.setLevel(log_level)
        self.print_output    = print_output
        
        self.cpus_per_worker = cpus_per_worker
        self.command         = command
        self.task_keys       = task_keys
        self.task_values     = task_values
        
        # MPI setup
        self.comm      = comm
        self.size      = comm.size
        self.rank      = comm.rank
        self.pool_comm = None
        
        # the parser
        self.parser = rsdfit_parser()
                
        # initialize a special string formatter
        self.formatter = string.Formatter()
        self.formatter.parse = lambda l: lib.MyStringParse(self.formatter, l, self.task_keys)
        
        # compute the batch params from the command
        self.batch_params = lib.find_batch_parameters(self.command.config, self.task_keys)
                
    def task_kwargs(self, itask):
        """
        The dictionary of parameters to update for each task
        """
        return dict(zip(self.task_keys, self.task_values[itask]))
        
    def initialize_driver(self):
        """
        Initialize the `RSDFitDriver` object on all ranks
        """
        # update with first value
        itask = 0

        # master will parse the args
        this_config = None; rsdfit_args = None
        if self.comm.rank == 0:
                  
            # get the kwargs
            kwargs = self.task_kwargs(itask)
            
            # this writes out the param file
            with self.command.update(kwargs, self.formatter) as command:
                this_config = command.param_file
                self.logger.debug("creating temporary file: %s" %this_config)
                rsdfit_args = command.args
        
        # bcast the file name to all in the worker pool
        this_config = self.comm.bcast(this_config, root=0)
        rsdfit_args = self.comm.bcast(rsdfit_args, root=0)
        self.temp_config = this_config

        # get the args
        self.logger.debug("calling rsdfit with arguments: %s" %str(rsdfit_args))
        
        args = None
        if self.comm.size > 1:
            if self.comm.rank == 0:
                args = self.parser.parse_args(rsdfit_args)
            args = self.comm.bcast(args, root=0)
        else:
            args = self.parser.parse_args(rsdfit_args)
                      
        # load the driver for everyone but root 
        if self.comm.rank != 0:     
            args = vars(args)
            mode = args.pop('subparser_name')
            self.driver = rsdfit.RSDFitDriver(self.pool_comm, mode, **args)
                    
    def initialize_pool_comm(self):
        """
        Internal function that initializes the `MPI.Intracomm` used by the 
        pool of workers. This will be passed to the task function and used 
        in task computation
        """
        # split the ranks
        self.pool_comm = None
        chain_ranks = []
        color = 0
        total_ranks = 0
        i = 0
        for i, ranks in split_ranks(self.size, self.cpus_per_worker):
            chain_ranks.append(ranks[0])
            if self.rank in ranks: color = i+1
            total_ranks += len(ranks)
        
        self.workers = i+1 # store the total number of workers
        leftover= (self.size - 1) - total_ranks
        if leftover and self.rank == 0:
            args = (self.cpus_per_worker, self.size-1, leftover)
            self.logger.warning("with `cpus_per_worker` = %d and %d available ranks, %d ranks will do no work" %args)
            
        # crash if we only have one process or one worker
        if self.size <= self.workers:
            args = (self.size, self.workers+1, self.workers)
            raise ValueError("only have %d ranks; need at least %d to use the desired %d workers" %args)
            
        # ranks that will do work have a nonzero color now
        self._valid_worker = color > 0
        
        # track how many tasks each worker does
        if self._valid_worker:
            self._completed_tasks = 0
        
        # split the comm between the workers
        self.pool_comm = self.comm.Split(color, 0)
                
    def run_all(self):
        """
        Run all of the tasks
        """    
        # just print the output directories and return
        if self.print_output:
            if self.rank == 0:
                for i in range(len(self.task_values)):
                    self._print_output(i)
                return 
            else:
                return 
        
        
        # define MPI message tags
        tags = enum('READY', 'DONE', 'EXIT', 'START')
        status = MPI.Status()
         
        try:
            # make the pool comm
            self.initialize_pool_comm()
    
            # the total numbe rof tasks
            num_tasks = len(self.task_values)
            
            # initialize the driver for everyone but master
            self.initialize_driver()
    
            # master distributes the tasks
            if self.rank == 0:
        
                # initialize
                task_index = 0
                closed_workers = 0
        
                # loop until all workers have finished with no more tasks
                self.logger.info("master starting with %d worker(s) with %d total tasks" %(self.workers, num_tasks))
                while closed_workers < self.workers:
                    data = self.comm.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)
                    source = status.Get_source()
                    tag = status.Get_tag()
            
                    # worker is ready, so send it a task
                    if tag == tags.READY:
                        if task_index < num_tasks:
                            self.comm.send(task_index, dest=source, tag=tags.START)
                            self.logger.info("sending task `%s` to worker %d" %(str(self.task_values[task_index]), source))
                            task_index += 1
                        else:
                            self.comm.send(None, dest=source, tag=tags.EXIT)
                    elif tag == tags.DONE:
                        results = data
                        self.logger.debug("received result from worker %d" %source)
                    elif tag == tags.EXIT:
                        closed_workers += 1
                        self.logger.debug("worker %d has exited, closed workers = %d" %(source, closed_workers))
    
            # worker processes wait and execute single jobs
            elif self._valid_worker:
                if self.pool_comm.rank == 0:
                    args = (self.rank, MPI.Get_processor_name(), self.pool_comm.size)
                    self.logger.info("pool master rank is %d on %s with %d processes available" %args)
                while True:
                    itask = -1
                    tag = -1
        
                    # have the master rank of the pool ask for task and then broadcast
                    if self.pool_comm.rank == 0:
                        self.comm.send(None, dest=0, tag=tags.READY)
                        itask = self.comm.recv(source=0, tag=MPI.ANY_TAG, status=status)
                        tag = status.Get_tag()
                    itask = self.pool_comm.bcast(itask)
                    tag = self.pool_comm.bcast(tag)
        
                    # do the work here
                    if tag == tags.START:
                        if self.print_output:
                            result = self._print_output(itask)
                        else:
                            result = self.run_rsdfit(itask)
                        self.pool_comm.Barrier() # wait for everyone
                        if self.pool_comm.rank == 0:
                            self.comm.send(result, dest=0, tag=tags.DONE) # done this task
                    elif tag == tags.EXIT:
                        break

                self.pool_comm.Barrier()
                if self.pool_comm.rank == 0:
                    self.comm.send(None, dest=0, tag=tags.EXIT) # exiting
        except Exception as e:
            self.logger.error("an exception has occurred on one of the ranks...all ranks exiting")
            self.logger.error(traceback.format_exc())
            
            # bit of hack that forces mpi4py to exit all ranks
            # see https://groups.google.com/forum/embed/#!topic/mpi4py/RovYzJ8qkbc
            os._exit(1)
        
        finally:
            # free and exit
            self.logger.debug("rank %d process finished" %self.rank)
            self.comm.Barrier()
            
            if self.rank == 0:
                self.logger.info("master is finished; terminating")
                if self.pool_comm is not None:
                    self.pool_comm.Free()
          
                if os.path.exists(self.temp_config): 
                    self.logger.debug("removing temporary file: %s" %self.temp_config)
                    os.remove(self.temp_config)
    
    def _print_output(self, itask):
        """
        Just print the output
        """
        # get the kwargs for this task
        kwargs = self.task_kwargs(itask)
        
        # update the attributes of the RSDFitDriver
        with self.command.update(kwargs, self.formatter) as command:
            
            # just print the output and return
            print(command.output_dir)
            if os.path.exists(command.param_file):
               os.remove(command.param_file)
            return
    
    def run_rsdfit(self, itask):
        """
        Run the algorithm once, using the parameters specified for this task
        iteration specified by `itask`
    
        Parameters
        ----------
        itask : int
            the integer index of this task
        """
        # initialization
        update_data = False; update_model = False
        fit_params = self.driver.algorithm.theory.fit_params
        
        # get the kwargs for this task
        kwargs = self.task_kwargs(itask)
        
        # update the attributes of the RSDFitDriver
        with self.command.update(kwargs, self.formatter) as command:
            args = vars(self.parser.parse_args(command.args))
            for k in args:
                setattr(self.driver, k, args[k])
            
            # update data, driver, and theory values
            for p in self.batch_params:

                # the updated value
                v = rgetattr(getattr(command, p.key), p.subkey)
                            
                # driver param
                if p.key == 'driver':
                    self.driver.algorithm.params.add(p.subkey, value=v)
                # data param
                elif p.key == 'data':
                    self.driver.algorithm.data.params.add(p.subkey, value=v)
                    update_data = True
                # theory param
                elif p.key == 'theory':
                    
                    name = p.subkey.split('.')[0]
                    if name not in fit_params:
                        raise ValueError("cannot update value of theory parameter '%s'; does not exist" %name)
                    
                    fit_params[name].value = v
                    fit_params[name].fiducial = v
                    update_model = True
                        
        # update the data?
        if update_data:
            self.driver.algorithm.data.initialize()
            
        # update the model?        
        if update_model:
            fit_params.update_values()
            self.driver.algorithm.theory.update_model()
        
        # write the parameters to file
        filename = os.path.join(self.driver.folder, params_filename)
        self.driver.algorithm.to_file(filename)
            
        # okay, now run
        self.driver.run()
