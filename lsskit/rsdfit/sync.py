from paramiko import SSHClient, SSHConfig
import os

from . import RSDFIT, RSDFIT_MODELS, RSDFIT_DATA, RSDFIT_FITS

RSYNC = "rsync -e ssh -avzl --progress --delete"

class NERSCConnection(object):
    """
    Context manager to connect to NERSC via ``ssh``
    """
    def __init__(self, host):
        
        # store the host
        self.host = host
        
        # read the ssh config file
        config = SSHConfig()
        fname = os.path.join(os.environ['HOME'], '.ssh', 'config')
        config.parse(open(fname))
        self.config = config.lookup(host)
        
        
    def run(self, cmd):
        """
        Run a command on NERSC remotely via ssh; this first 
        source the bash profile on NERSC before executing any
        commands
        
        Returns
        -------
        str:
            the stdout return value
        """
        
        cmd = "source ~/.bash_profile; " + cmd
        stdin, stdout, stderr = self.ssh_client.exec_command(cmd)
        return stdout.read().rstrip()
    
    def __enter__(self):
        """
        Use paramiko to connect to NERSC over ``ssh``
        """
        # ssh client
        self.ssh_client = SSHClient()
        self.ssh_client.load_system_host_keys()
        
        # connect
        kws = {'username':self.config['user'], 'key_filename':self.config['identityfile']}
        self.ssh_client.connect(self.config['hostname'], **kws)
        
        return self
        
    def __exit__ (self, exc_type, exc_value, traceback):
        """
        Close the ssh client when exiting the context
        """
        self.ssh_client.close()
        
def sync_models(host, dry_run=False):
    """
    Sync the RSD models to NERSC
    
    Parameters
    ----------
    host : {'cori', 'edison'}
        the name of the remote host
    dry_run : bool, optional
        whether to do a dry-run
    """
    # get the models directory
    with NERSCConnection(host) as nersc:
        remote_dir = nersc.run("python -c 'from lsskit import rsdfit; print rsdfit.RSDFIT_MODELS'")

    # the command + options 
    cmd = RSYNC
    if dry_run: cmd += ' --dry-run'
    cmd += " --exclude='*.py'"
    
    # add the directories and run the command
    cmd += " %s nhand@%s:%s" %(RSDFIT_MODELS, host, remote_dir)
    ret = os.system(cmd)
    

def sync_data(host, dry_run=False):
    """
    Sync the RSDFit ``data`` directory to NERSC
    
    Parameters
    ----------
    host : {'cori', 'edison'}
        the name of the remote host
    dry_run : bool, optional
        whether to do a dry-run
    """
    # get the data directory
    with NERSCConnection(host) as nersc:
        remote_dir = nersc.run("python -c 'from lsskit import rsdfit; print rsdfit.RSDFIT_DATA'")

    # the command + options 
    cmd = RSYNC
    if dry_run: cmd += ' --dry-run'
    cmd += " --exclude='.*'"
    
    # add the directories and run the command
    cmd += " %s nhand@%s:%s" %(RSDFIT_DATA, host, remote_dir)
    ret = os.system(cmd)
        
def sync_params(host, dry_run=False):
    """
    Sync the RSDFit ``params`` directory to NERSC
    
    Parameters
    ----------
    host : {'cori', 'edison'}
        the name of the remote host
    dry_run : bool, optional
        whether to do a dry-run
    """
    # get the data directory
    with NERSCConnection(host) as nersc:
        remote_dir = nersc.run("python -c 'from lsskit import rsdfit; print rsdfit.RSDFIT'")
        
    # the command + options 
    cmd = RSYNC
    if dry_run: cmd += ' --dry-run'
    cmd += " --exclude='.*'"
    
    # add the directories and run the command
    cmd += " %s/params/ nhand@%s:%s/params" %(RSDFIT, host, remote_dir)
    ret = os.system(cmd)
    
def sync_run(host, dry_run=False):
    """
    Sync the RSDFit ``run`` directory to NERSC
    
    Parameters
    ----------
    host : {'cori', 'edison'}
        the name of the remote host
    dry_run : bool, optional
        whether to do a dry-run
    """
    # get the data directory
    with NERSCConnection(host) as nersc:
        remote_dir = nersc.run("python -c 'from lsskit import rsdfit; print rsdfit.RSDFIT'")

    # the command + options 
    cmd = RSYNC
    if dry_run: cmd += ' --dry-run'
    cmd += " --exclude='.*' --exclude='output'"
    
    # add the directories and run the command
    cmd += " %s/run/ nhand@%s:%s/run" %(RSDFIT, host, remote_dir)
    ret = os.system(cmd)

def sync_fits(direction, host, path=None, dry_run=False):
    """
    Sync the RSD fits to/from NERSC
    
    Parameters
    ----------
    direction : {'to', 'from'}
        the direction to do the transfer, either 'to' or 'from'
        the remote host
    host : {'cori', 'edison'}
        the name of the remote host
    path : str
        a subpath from the ``RSDFIT_FITS`` directory
    dry_run : bool, optional
        whether to do a dry-run
    """
    # get the fits directory
    with NERSCConnection(host) as nersc:
        remote_dir = nersc.run("python -c 'from lsskit import rsdfit; print rsdfit.RSDFIT_FITS'")

    # the command + options 
    cmd = [RSYNC.split('--delete')[0]]
    cmd += ["--exclude='info'", "--exclude='plots'", "--exclude='.*'"]
    if dry_run: cmd.append('--dry-run')
    
    # add the directories and run the command
    dirs = [RSDFIT_FITS, "nhand@%s:%s" %(host, remote_dir)]
    if path is not None:
        dirs[0] += path
        dirs[1] += path
            
    if direction == 'from':
        dirs = dirs[::-1]
    cmd += dirs    
    
    ret = os.system(" ".join(cmd))