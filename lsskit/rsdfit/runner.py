import argparse
import os
import textwrap as tw
import sys
import subprocess
import time
import tempfile

from . import RSDFIT_FITS

def get_modified_time(o):
    return "last modified: %s" % time.ctime(os.path.getmtime(o))
    
def format_output(o):
    
    relpath = os.path.relpath(o, RSDFIT_FITS)
    return os.path.join("$RSDFIT_FITS", relpath)
        
class OutputAction(argparse.Action):
    """
    Action similar to ``help`` to print out 
    the output directories and the last modified times
    """
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(OutputAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        
        print "Output directories\n" + "-"*20
        for i, command in enumerate(RSDFitRunner.commands):
            
            command += " --output"
            
            # print test number first
            toprint = "%d:\n" %i
            
            # get the output directories
            output = RSDFitRunner._execute(command, output_only=True)
            for o in output: 
                if 'warning' not in o:
                    o_ = format_output(o)
                else:
                    o_ = "warning: no output directory information"

                s = " "*4 + o_ + "\n"
                if os.path.isdir(o):
                    s += " "*8 + get_modified_time(o) + '\n'
                elif 'warning' not in o:
                    s += " "*8 + "does not exist yet" + '\n'
                else:
                    s += " "*8 + o[8:] + '\n'
                toprint += s
                
            print toprint
            
        parser.exit()

class InfoAction(argparse.Action):
    """
    Action similar to ``help`` to print out 
    the various registered commands for the ``RSDFitRunner``
    class
    """
    def __init__(self,
                 option_strings,
                 dest=argparse.SUPPRESS,
                 default=argparse.SUPPRESS,
                 help=None):
        super(InfoAction, self).__init__(
            option_strings=option_strings,
            dest=dest,
            default=default,
            nargs=0,
            help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        
        print "Registered commands\n" + "-"*20
        for i, command in enumerate(RSDFitRunner.commands):
            c = tw.dedent(command).strip()
            c = tw.fill(c, initial_indent=' '*4, subsequent_indent=' '*4, width=80)
            print "%d:\n%s\n" %(i, c)
            
        parser.exit()

class RSDFitRunner(object):
    """
    Class to run ``rsdfit`` commands
    """
    commands = []
    
    @classmethod
    def register(cls, command):
        """
        Register a new ``rsdfit`` command
        """
        cls.commands.append(command)
        
    @classmethod
    def execute(cls):
        """
        Execute the ``RSDFitRunner`` command
        """
        # parse and get the command
        ns = cls.parse_args()
        
        # append any NERSC-related options
        command = cls.commands[ns.testno]
        if ns.nodes is not None and ns.parition is not None:
            print "submitting job: requesting %d nodes on '%s' queue" %(ns.nodes, ns.partition)
            command += " -N %d -p %s" %(ns.nodes, ns.partition)
        
        # execute
        cls._execute(, clean=ns.clean)
            
    @classmethod
    def _execute(cls, command, output_only=False, clean=False):
        """
        Internal function to execute the command
        """        
        # extract the relevant output directories
        if output_only or clean:
            
            if "--output" not in command:
                command += " --output"
                
            with tempfile.TemporaryFile() as stderr:
                try:
                    out = subprocess.check_output(command, shell=True, stderr=stderr)
                except subprocess.CalledProcessError as e:
                    stderr.seek(0)
                    error = stderr.read()
                    if len(error):
                        
                        if clean:
                            raise ValueError("cannot clean specified output directory due to exception")
                        
                        if '`start_from`' in error:
                            return ["warning: `start_from` variable does not currently exist"]
                        elif "the input configuration file does not exist" in error:
                            return ["warning: the specified configuration file does not exist"]
                        else:
                            indent = " "*14
                            error = error.split("\n")
                            error = "\n".join([indent + l for l in error])
                            return ["warning: unknown exception raised:\n%s" %error]
                            
            dirs = [o for o in out.split("\n") if o]
            if not clean:
                return dirs
            else:
                for d in dirs:
                    if os.path.isdir(d):
                        p = os.path.join(d, '*')
                        os.system("rm -i -r %s" %p)
                
            
        # just run the command
        else:
        
            # print the command
            c = tw.dedent(command).strip()
            c = tw.fill(c, initial_indent=' '*4, subsequent_indent=' '*4, width=80)
            print "executing:\n%s" %c
        
            # execute
            os.system(command)

    @classmethod
    def parse_args(cls):
        """
        Parse the command-line arguments
        """
        desc = "run a ``rsdfit`` command from a set of registered commands"
        parser = argparse.ArgumentParser(description=desc)
        
        h = 'the integer number of the test to run'
        parser.add_argument('testno', type=int, help=h)
        
        h = 'print out the various commands'
        parser.add_argument('-i', '--info', action=InfoAction, help=h)
        
        h = 'print out the output directories and last modified timees for each registerd command'
        parser.add_argument('-o', '--output', action=OutputAction, help=h)
        
        h = 'remove all files from the specified output directory'
        parser.add_argument('--clean', action='store_true', help=h)
        
        h = 'the number of nodes to request'
        parser.add_argument('-N', '--nodes', type=int, help=h)
        
        h = 'the NERSC partition to submit to'
        parser.add_argument('-p', '--partition', type=str, help=h)
        
        ns = parser.parse_args()
        
        # make sure the integer value is valid
        if not (0 <= ns.testno < len(cls.commands)):
            N = len(cls.commands)
            raise ValueError("input ``testno`` must be between [0, %d]" %N)
            
        return ns