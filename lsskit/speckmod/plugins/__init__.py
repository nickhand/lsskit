import lsskit

#------------------------------------------------------------------------------
class ModelInput:
    """
    Mount point for model plugins
    """
    __metaclass__ = lsskit.PluginMount
    name = None
    plugin_type = None
    
    from argparse import ArgumentParser
    parser = ArgumentParser("", add_help=False)
    subparsers = parser.add_subparsers()

    def __init__(self, dict):
        self.__dict__.update(dict)

    def __str__(self):
        return self.string
        
    @classmethod
    def parse(cls, string): 
        words = string.split(':')
        
        ns = cls.parser.parse_args(words)
        klass = ns.klass
        d = ns.__dict__
        del d['klass']
        d['string'] = string
        model = klass(d)
        return model

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string   

    @classmethod
    def add_parser(cls, name, usage):
        return cls.subparsers.add_parser(name, 
                usage=usage, add_help=False)
    
    @classmethod
    def format_help(cls, plugin_type):
        rt = []
        for plugin in cls.plugins:
            if plugin_type != plugin.plugin_type:
                continue
            k = plugin.name
            rt.append(cls.subparsers.choices[k].format_help())

        if not len(rt):
            return "No available input %s plugins" %plugin_type
        else:
            return '\n'.join(rt)
            
#------------------------------------------------------------------------------
import contextlib

class ModelResultsStorage:        
    __metaclass__ = lsskit.PluginMount
    name = None
    plugin_type = None

    from argparse import ArgumentParser
    parser = ArgumentParser("", add_help=False)
    subparsers = parser.add_subparsers()

    def __init__(self, dict):
        self.__dict__.update(dict)

    def __str__(self):
        return self.string
        
    @classmethod
    def parse(cls, string): 
        words = string.split(':')
    
        ns = cls.parser.parse_args(words)
        klass = ns.klass
        d = ns.__dict__
        del d['klass']
        d['string'] = string
        model = klass(d)
        return model

    def __eq__(self, other):
        return self.string == other.string

    def __ne__(self, other):
        return self.string != other.string   

    @classmethod
    def add_parser(cls, name, usage):
        return cls.subparsers.add_parser(name, 
                usage=usage, add_help=False)

    @classmethod
    def format_help(cls, plugin_type):
        rt = []
        for plugin in cls.plugins:
            if plugin_type != plugin.plugin_type:
                continue
            k = plugin.name
            rt.append(cls.subparsers.choices[k].format_help())

        if not len(rt):
            return "No available input %s plugins" %plugin_type
        else:
            return '\n'.join(rt)
    
    @contextlib.contextmanager
    def open(self):
        output = self.__open__()
        try:
            yield output
        finally:
            self.__finalize__()
            
    def write(self, key, result):
        return NotImplemented
            
#------------------------------------------------------------------------------      


builtins = ['RunPBModelData', 'PhmResidualPadeModel', 'LambdaLogModel', 'PhmCorrectedPTModel', 'PhhModel',
            'LambdaPadeModel', 'BestfitParamStorage', 'BestfitFunctionStorage', 'InterpolationStorage']
for plugin in builtins:
    filename = lsskit.path.join(lsskit.path.dirname(__file__), plugin + '.py')
    globals().update(lsskit.load(filename))
 