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
  
  
def parse_args(desc, samples):
    """
    Parse the command line arguments and return the namespace
    
    Parameters
    ----------
    desc : str
        the description for the argument parser
    samples : list of str
        a list of the samples names. these provide the choices
        for the `sample` command line argument, which selects
        which sample we are computing results for
    
    Returns
    -------
    args : argparse.Namespace
        namespace holding the commandline arguments
    """
    coords = 
    
    SamplesAction.valid = samples + ['all']
    
    parser = ap.ArgumentParser(description=desc, 
                formatter_class=ap.ArgumentDefaultsHelpFormatter)
                            
    h = 'the name of the PBS job file to run. This file should take one' + \
        'command line argument specfying the input `run_rsdfit.py` config parameter file'
    parser.add_argument('job_file', type=str, help=h)
    h = 'the sample name(s), must be one of the coord values'
    parser.add_argument('samples', nargs="+", action=SamplesAction, help=h)
    h = 'the name of the file specifying the main `run_rsdfit.py` parameters'
    parser.add_argument('-p', '--config', required=True, type=param_file, help=h)
    h = 'the name of the file specifying the selection parameters'
    parser.add_argument('-s', '--select', required=True, type=extra_iter_values, help=h)
    
    return parser.parse_args()

def qsub_samples(args, samples):
    """
    Submit the job script specified on the command line for the desired 
    sample(s). This could submit several job at once, but will 
    wait 1 second in between doing so.  
    """
    import subprocess
    import time
    
    # determine the sample we are running
    if len(args.samples) == 1 and args.samples[0] == 'all':
        args.samples = samples
        
    # initialize a string formatter
    formatter = string.Formatter()
    
    # submit the jobs
    for sample in args.samples:
        
        kwargs = {}
        if sample in args.select:
            kwargs = args.select[sample]
        formatter.parse = lambda l: my_string_parse(formatter, l, kwargs.keys())
    
        all_kwargs = [kw for _, kw, _, _ in args.config._formatter_parser() if kw]
        with tempfile.NamedTemporaryFile(delete=False) as ff:
            fname = ff.name
            valid = {k:v for k,v in kwargs.iteritems() if k in all_kwargs}
            ff.write(formatter.format(args.config, **valid))
        
        v_value = 'param_file=%s' %fname
        ret = subprocess.call(['qsub', '-v', v_value, args.job_file])
        time.sleep(1)

    
        
    
    
    
    