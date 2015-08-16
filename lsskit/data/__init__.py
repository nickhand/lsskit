import lsskit

class PowerSpectraLoader:
    __metaclass__ = lsskit.PluginMount

    name = None
    classes = {}

    def __init__(self, root):
        self.root = root

    @classmethod
    def store_class(cls, klass):
        cls.classes[klass.name] = klass

    @classmethod
    def get(cls, name, root_dir, plugin_path=None, **kwargs):
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
            lsskit.load(plugin_path)
            
        if name not in cls.classes:
            if len(cls.classes):
                valid = ", ".join("`%s`" %x.name for x in cls.classes.values())
                raise lsskit.PluginError("valid PowerSpectraLoader plugin names are: %s" %valid)
            else:
                raise lsskit.PluginError("no valid PowerSpectraLoader plugins registered")
            
        return cls.classes[name](root_dir, **kwargs)
            
builtins = ['RunPBHalo', 'RunPBGalaxy', 'TeppeiSims']
for plugin in builtins:
    filename = lsskit.path.join(lsskit.path.dirname(__file__), plugin + '.py')
    globals().update(lsskit.load(filename))
 