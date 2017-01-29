import numpy
import os.path as path
import warnings
warnings.filterwarnings("ignore",category=DeprecationWarning)

"""
    Declare PluginMount and various extention points.

    To define a Plugin, set __metaclass__ to PluginMount, and
    define a .register member.

"""
class PluginMount(type):
    
    def __init__(cls, name, bases, attrs):

        # only executes when processing the mount point itself.
        if not hasattr(cls, 'plugins'):
            cls.plugins = []
        # called for each plugin, which already has 'plugins' list
        else:
            # track names of classes
            cls.plugins.append(cls)
            
            # try to call register class method
            if hasattr(cls, 'register'):
                cls.register()

class PluginError(Exception):
    pass
    
def load(filename, namespace=None):
    """ An adapter for ArgumentParser to load a plugin.
        
        Parameters
        ----------
        filename : string
            path to the .py file
        namespace : dict
            global namespace, if None, an empty space will be created
        
        Returns
        -------
        namespace : dict
            modified global namespace of the plugin script.
    """
    if namespace is None:
        namespace = {}
    if path.isdir(filename):
        raise PluginError("can not `load` directory as plugin")
    try:
        with open(filename) as ff:
            code = compile(ff.read(), filename, "exec")
            exec(code, namespace)
    except Exception as e:
        raise PluginError("failed to load plugin '%s': %s" % (filename, str(e)))
    return namespace
