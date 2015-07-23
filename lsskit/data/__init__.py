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

class PluginException(Exception):
    pass


class PowerSpectraLoader:
    __metaclass__ = PluginMount

    name = None
    classes = {}

    def __init__(self, root):
        self.root = root

    @classmethod
    def store_class(cls, klass):
        cls.classes[klass.name] = klass

    @classmethod
    def get(cls, name, root_dir, plugin_path=None):
        """
        Return the PowerSpectraLoader plugin with
        the name ``name``, optionally loading it from the 
        path specified.
        
        Parameters
        ----------
        name : str  
            the registered name of the plugin
        root_dir : str
            the name of the root directory holding the results. This 
            argument is passed to the plugin initialization
        plugin_path : str, optional
            the path of the plugin file to load 
        """
        if plugin_path is not None:
            load(plugin_path)
            
        if name not in cls.classes:
            if len(cls.classes):
                valid = ", ".join("`%s`" %x.name for x in cls.classes.values())
                raise PluginException("valid PowerSpectraLoader plugin names are: %s" %valid)
            else:
                raise PluginException("no valid PowerSpectraLoader plugins registered")
            
        return cls.classes[name](root_dir)
            
#------------------------------------------------------------------------------          
import os.path
import sys
    
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
    if os.path.isdir(filename):
        raise ValueError("Can not load directory")
    try:
        execfile(filename, namespace)
    except Exception as e:
        raise PluginException("Failed to load plugin '%s': %s" % (filename, str(e)))
    return namespace
    
builtins = ['RunPB']
for plugin in builtins:
    globals().update(load(os.path.join(os.path.dirname(__file__), plugin + '.py')))
 