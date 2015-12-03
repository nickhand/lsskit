import argparse as ap
import os
import string
import tempfile

def my_string_parse(formatter, s, keys):
    l = list(string.Formatter.parse(formatter, s))
    toret = []
    for x in l:
        if x[1] in keys:
            toret.append(x)
        else:
            val = x[0]
            if x[1] is not None:
                fmt = "" if not x[2] else ":%s" %x[2]
                val += "{%s%s}" %(x[1], fmt)
            toret.append((val, None, None, None))
    return iter(toret)

def param_file(s):
    """
    A file holding parameters, which will be read, and returned
    as a string
    """
    if not os.path.exists(s):
        raise RuntimeError("file `%s` does not exist" %s)
    return open(s, 'r').read()

def extra_iter_values(s):
    """
    Provided an existing file name, read the file into 
    a dictionary. The keys are interpreted as the string format name, 
    and the values are a list of values to iterate over for each job
    """
    if not os.path.exists(s):
        raise RuntimeError("file `%s` does not exist" %s)
    toret = {}
    execfile(s, globals(), toret)
    return toret
  
  
def parse_args(desc, dims, coords):
    """
    Parse the command line arguments and return the namespace. 
    Extra options valid for ``rsdfit`` can be passed here, 
    and will be treated as ``unknown`` arguments
    
    Parameters
    ----------
    desc : str
        the description for the argument parser
    dims : list of str
        the list of the dimensions for the samples
    coords : list of str
        the values corresponding to the sample dimensions
    
    Returns
    -------
    args : argparse.Namespace
        namespace holding the commandline arguments
    """
    parser = ap.ArgumentParser(description=desc)
                            
    h = 'the name of the batch job script to run. This file should take one' + \
        'command line argument `command` which is the rsdfit command to run'
    parser.add_argument('job_file', type=str, help=h)
    h = 'the name of the file specifying the template file for configuration parameters'
    parser.add_argument('-p', '--config', required=True, type=param_file, help=h)
    h = 'the name of the file specifying the selection parameters'
    parser.add_argument('-s', '--select', required=True, type=extra_iter_values, help=h)
    h = 'the path of python script that will call `run_rsdfit`, and store the ' + \
        'run command in a variable called `command`; it should use the `param_file` variable'
    parser.add_argument('--setup', required=True, type=str, help=h)
    h = 'just call the command using os.system, instead of submitting a batch job'
    parser.add_argument('--call', action='store_true', help=h)
    h = 'the job submission mode'
    parser.add_argument('--mode', choices=['pbs', 'slurm'], default='pbs', help=h)
    
    # add the samples
    for i, (dim, vals) in enumerate(zip(dims, coords)):
        h = 'the #%d sample dimension' %i
        parser.add_argument('--%s' %dim, nargs='+', choices=['all']+vals, help=h, required=True)
    
    return parser.parse_known_args()

def submit_jobs(args, dims, coords, rsdfit_options=[], mode='pbs'):
    """
    Submit the job script specified on the command line for the desired 
    sample(s). This could submit several job at once, but will 
    wait 1 second in between doing so.
    
    This executes the script specified by ``args.setup``, which calls
    ``run_rsdfit`` and defines the ``rsdfit`` command which will be
    passed to the job script as the ``command`` environment variable
    
    Notes
    -----
    *   given the input template config file, each job will string format
        the template file using the dictionary of values stored
        in the select file, i.e. for `box_A` iteration, it reads the
        ``box_A`` dict from the select file and string formats using
        those key/value pairs
    *   this creates a temporary file that is not deleted 
    
    Parameters
    ----------
    args : argparse.Namespace
        a namespace holding the parsed arguments from ``parse_args``
    dims : list
        the list of dimension names to iterate over, i.e, ``['box', 'kmax']``
    coords : list
        the list of iteration values for each dimension, i.e., ``[['A', 'B'], ['02', '03']]``
    rsdfit_options : list, optional
        a list of optional keywords that will be passed to the ``rsdfit`` script
    """
    import time
    import itertools
    
    if mode not in ['pbs', 'slurm']:
        raise ValueError("``mode`` must be `pbs` or `slurm`")
    
    # initialize a special string formatter
    formatter = string.Formatter()
    
    # determine the samples we want to run
    samples = []
    for i, dim in enumerate(dims):
        val = getattr(args, dim)
        if len(val) == 1 and val[0] == 'all':
            val = coords[i]
        samples.append(val)
            
    # loop over each sample iteration
    for sample in itertools.product(*samples):
        
        # grab the kwargs to format for each dimension
        kwargs = {}
        for i, dim in enumerate(dims):
            name = '%s_%s' %(dim, sample[i])
            if name in args.select:
                kwargs.update(args.select[name])
        formatter.parse = lambda l: my_string_parse(formatter, l, kwargs.keys())
    
        # write out the formatted config file for this iteration
        all_kwargs = [kw for _, kw, _, _ in args.config._formatter_parser() if kw]
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            fname = ff.name
            valid = {k:v for k,v in kwargs.iteritems() if k in all_kwargs}
            ff.write(formatter.format(args.config, **valid))
        
        # call the python script that takes the `param_file` variable
        # this should call run_rsdfit and set the corresponding return
        # value to the variable ``command``
        call = {'param_file' : fname, 'rsdfit_options' : rsdfit_options}
        execfile(args.setup, globals(), call)
        if 'command' not in call:
            raise RuntimeError("python setup script for run_rsdfit should define the `command` variable")
        
        command = call['command']
        
        # submit a job, using qsub or slurm
        if not args.call:
            
            command_str = 'command=%s' %command
            if mode == 'pbs':
                ret = os.system("qsub -v '%s' %s" %(command_str, args.job_file))
            else:
                ret = os.system("sbatch '--export=%s=%s,ALL'" %(command_str, args.job_file))
            time.sleep(1)
        # just run the command
        else:
            os.system(command)

    
        
    
    
    
    