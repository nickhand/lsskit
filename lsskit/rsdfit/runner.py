import argparse
import os
import textwrap as tw
import sys
import subprocess
import time
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
                o_ = format_output(o)
                s = " "*4 + o_ + "\n"
                if os.path.isdir(o):
                    s += " "*8 + get_modified_time(o) + '\n'
                else:
                    s += " "*8 + "does not exist yet" + '\n'
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
        cls._execute(cls.commands[ns.testno])
            
    @classmethod
    def _execute(cls, command, output_only=False):
        """
        Internal function to execute the command
        """        
        # extract the relevant output directories
        if output_only:
            with open(os.devnull, 'w') as FNULL:
                out = subprocess.check_output(command, shell=True, stderr=FNULL)
            return [o for o in out.split("\n") if o]
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
        parser.add_argument('-r', '--report', dest='outpu', action=OutputAction, help=h)
        
        ns = parser.parse_args()
        
        # make sure the integer value is valid
        if not (0 <= ns.testno < len(cls.commands)):
            N = len(cls.commands)
            raise ValueError("input ``testno`` must be between [0, %d]" %N)
            
        return ns