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
    Parse the command line arguments and return the namespace
    
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
                            
    h = 'the name of the PBS job file to run. This file should take one' + \
        'command line argument specfying the input `run_rsdfit.py` config parameter file'
    parser.add_argument('job_file', type=str, help=h)
    h = 'the name of the file specifying the main `run_rsdfit.py` parameters'
    parser.add_argument('-p', '--config', required=True, type=param_file, help=h)
    h = 'the name of the file specifying the selection parameters'
    parser.add_argument('-s', '--select', required=True, type=extra_iter_values, help=h)
    
    # add the samples
    for i, (dim, vals) in enumerate(zip(dims, coords)):
        h = 'the #%d sample dimension' %i
        parser.add_argument('--%s' %dim, nargs='+', choices=['all']+vals, help=h, required=True)
    
    return parser.parse_args()

def qsub_samples(args, dims, coords):
    """
    Submit the job script specified on the command line for the desired 
    sample(s). This could submit several job at once, but will 
    wait 1 second in between doing so.  
    """
    import subprocess
    import time
    import itertools
    
    # initialize a string formatter
    formatter = string.Formatter()
    
    samples = []
    for i, dim in enumerate(dims):
        val = getattr(args, dim)
        if len(val) == 1 and val[0] == 'all':
            val = coords[i]
        samples.append(val)
            
    # submit the jobs
    for sample in itertools.product(*samples):
        print sample
        kwargs = {}
        for i, dim in enumerate(dims):
            name = '%s_%s' %(dim, sample[i])
            if name in args.select:
                kwargs.update(args.select[name])
        formatter.parse = lambda l: my_string_parse(formatter, l, kwargs.keys())
    
        all_kwargs = [kw for _, kw, _, _ in args.config._formatter_parser() if kw]
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            fname = ff.name
            valid = {k:v for k,v in kwargs.iteritems() if k in all_kwargs}
            ff.write(formatter.format(args.config, **valid))
        print fname
        #v_value = 'param_file=%s' %fname
        #ret = subprocess.call(['qsub', '-v', v_value, args.job_file])
        time.sleep(1)

    
        
    
    
    
    