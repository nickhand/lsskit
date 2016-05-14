import argparse
import sys
import logging
import tempfile
from mpi4py import MPI

from lsskit.rsdfit import lib, batch, RSDFIT_BATCH
from lsskit.rsdfit.command import RSDFitBatchCommand

# setup the logging
rank = MPI.COMM_WORLD.rank
name = MPI.Get_processor_name()
logging.basicConfig(level=logging.DEBUG,
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
        
    h = 'the number of nodes to use when submitting the job'
    parser.add_argument('-N', '--nodes', type=int, help=h)

    h = 'the partition to submit the job to'
    parser.add_argument('-p', '--partition', type=str, choices=['debug', 'regular'], help=h)
    
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
    
    # parse known and unknown arguments
    ns, other = parser.parse_known_args()

    # get the tasks
    if ns.tasks is not None:
        tasks = open(ns.tasks, 'r').readlines()
    else:
        if MPI.COMM_WORLD.rank == 0:
            tasks = sys.stdin.readlines()
        else:
            tasks = None
        # respect the root rank stdin only;
        # on some systems, the stdin is only redirected to the root rank.
        tasks = MPI.COMM_WORLD.bcast(tasks)

    # submit the job, instead of running
    if ns.nodes is not None or ns.partition is not None:
        if ns.nodes is None or ns.partition is None:
            raise ValueError("both `nodes` and `partition` must be given to submit job")

        if MPI.COMM_WORLD.rank == 0:            
            args = sys.argv[1:]
        
            # remove the nodes and partition arguments
            i = args.index('-N') if '-N' in args else args.index('--nodes')
            j = args.index('-p') if '-p' in args else args.index('--partition')
            for ii in reversed([i, j]):
                args.pop(ii+1)
                args.pop(ii)
                    
            # write out the iterating values to a temp file
            with tempfile.NamedTemporaryFile(delete=False) as ff:
                fname = ff.name
                ff.write("".join(tasks))
            call_signature = [fname] + args
        
            # add the executable
            if ns.command is None: 
                ns.command = RSDFIT_BATCH
            call_signature = ns.command.split() + call_signature
        
            # submit the job and return
            command = " ".join(call_signature)
            lib.submit_rsdfit_job(command, ns.nodes, ns.partition)
        
        return 
        
    # make the command
    kws = {}
    kws['options'] = other
    kws['theory_options'] = ns.theory_options
    kws['tag'] = ns.tag
    kws['executable'] = ns.command
    command = RSDFitBatchCommand(ns.config_template, ns.stat, ns.kmax, **kws)
        
    # format the tasks
    task_values = [line.split() for line in tasks]
    task_keys = task_values[0]
    task_values = task_values[1:]

    # initialize the task manager
    args = (MPI.COMM_WORLD, ns.cpus_per_worker, command, task_keys, task_values)
    manager = batch.RSDFitBatch(*args, log_level=ns.log_level)
    
    # and run!
    manager.run_all()
    
if __name__ == '__main__':
    main()