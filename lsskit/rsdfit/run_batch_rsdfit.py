import argparse
import sys
import logging
import tempfile

from lsskit.rsdfit import lib, RSDFIT_BATCH
from lsskit.rsdfit.command import RSDFitBatchCommand


def add_console_logger():
    """
    Add a console logger
    """
    from mpi4py import MPI
    
    # setup the logging
    rank = MPI.COMM_WORLD.rank
    name = MPI.Get_processor_name()
    logging.basicConfig(level=logging.INFO,
                        format='rank %d on %s: '%(rank,name) + \
                                '%(asctime)s %(name)-15s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M')

def main():
    """
    Run the ``rsdfit`` algorithm in batch mode
    """
    desc = "run ``rsdfit`` in batch mode, iterating over several configurations, "
    desc += "possibly executed in parallel"
    parser = argparse.ArgumentParser(description=desc)
        
    # the tasks, possibly read from stdin
    h = "the tasks to loop over, either read from file or stdin"
    parser.add_argument('tasks', type=argparse.FileType(mode='r'), nargs='?', 
        default=None, help=h)
    
    # the number of independent workers
    h = """the desired number of ranks assigned to each independent
            worker, when iterating over the tasks in parallel""" 
    parser.add_argument('cpus_per_worker', type=int, help=h)
    
    h = 'the name of the file holding the template configuration parameters'
    parser.add_argument('config_template', type=str, help=h)
    
    h = 'the executable command to call, i.e, ``rsdfit`` or ``mpirun -n 2 rsdfit``'
    parser.add_argument('--command', type=str, help=h)

    h = "set the logging output to debug, with lots more info printed"
    parser.add_argument('--debug', help=h, action="store_const", dest="log_level", 
                        const=logging.DEBUG, default=logging.INFO)
        
    h = 'just print the output file and exit'
    parser.add_argument('-o', '--output', dest='print_output', action='store_true', help=h)
    
    # NERSC-related options
    nersc = parser.add_argument_group("NERSC-related options")
    
    h = 'the number of nodes to use when submitting the job'
    nersc.add_argument('-N', '--nodes', type=int, help=h)    
    
    h = 'the requested amount of time'
    nersc.add_argument('-t', '--time', type=lib.slurm_time, help=h)
    
    h = 'the partition to submit the job to'
    nersc.add_argument('-p', '--partition', type=str, choices=['debug', 'regular'], help=h)
    
    # required named arguments
    group = parser.add_argument_group('rsdfit configuration')
    
    h = 'the statistic; either pkmu or poles'
    group.add_argument('--stat', choices=['pkmu', 'poles'], required=True, help=h)
    
    h = 'the maximum k to use'
    group.add_argument('--kmax', type=float, nargs='*', required=True, help=h)
    
    h = 'additional options to apply the theory model, i.e., `mu_corr` or `so_corr`'
    group.add_argument('-th', '--theory_options', type=str, nargs='*', help=h)
    
    h = 'additional tag to append to the output directory'
    group.add_argument('--tag', type=str, default="", help=h)
    
    h = 'the directory to start from'
    group.add_argument('--start', type=str, help=h)
    
    # parse known and unknown arguments
    ns, other = parser.parse_known_args()

    # submit the job, instead of running
    if not ns.print_output and ns.nodes is not None or ns.partition is not None or ns.time is not None:
        if ns.nodes is None or ns.partition is None or ns.time is None:
            raise ValueError("`nodes`, `partition`, and `time` must all be given to submit job")

        args = sys.argv[1:]
    
        # get the tasks
        if ns.tasks is not None:
            tasks = ns.tasks.readlines()
        else:
            tasks = sys.stdin.readlines()
    
        # remove the nodes and partition arguments
        i = args.index('-N') if '-N' in args else args.index('--nodes')
        j = args.index('-p') if '-p' in args else args.index('--partition')
        k = args.index('-t') if '-t' in args else args.index('--time')
        for ii in reversed([i, j, k]):
            args.pop(ii+1)
            args.pop(ii)
                
        # write out the iterating values to a temp file
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            fname = ff.name
            ff.write(("".join(tasks)).encode())
        call_signature = [fname] + args
    
        # add the executable
        if ns.command is None: 
            ns.command = RSDFIT_BATCH
        call_signature = ns.command.split() + call_signature

        # submit the job and return
        command = " ".join(call_signature)
        lib.submit_rsdfit_job(command, ns.nodes, ns.partition, ns.time)
        
        return 
        
    else:
        from mpi4py import MPI
        from lsskit.rsdfit import batch
        
        # add the console logger
        add_console_logger()
        
        # get the tasks
        if ns.tasks is not None:
            tasks = ns.tasks.readlines()
        else:
            if MPI.COMM_WORLD.rank == 0:
                tasks = sys.stdin.readlines()
            else:
                tasks = None
            # respect the root rank stdin only;
            # on some systems, the stdin is only redirected to the root rank.
            tasks = MPI.COMM_WORLD.bcast(tasks)
        
        # make the command
        kws = {}
        kws['options'] = other
        kws['theory_options'] = ns.theory_options
        kws['tag'] = ns.tag
        kws['executable'] = ns.command
        kws['start_from'] = ns.start
        command = RSDFitBatchCommand(ns.config_template, ns.stat, ns.kmax, **kws)
        
        # format the tasks
        task_values = [line.split() for line in tasks]
        task_keys = task_values[0]
        task_values = task_values[1:]

        if len(tasks) == 1:
            ns.cpus_per_worker = MPI.COMM_WORLD.size
            
        # initialize the task manager
        args = (MPI.COMM_WORLD, ns.cpus_per_worker, command, task_keys, task_values)
        manager = batch.RSDFitBatch(*args, log_level=ns.log_level, print_output=ns.print_output)
    
        # and run!
        manager.run_all()
    
if __name__ == '__main__':
    main()