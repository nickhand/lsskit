from lsskit.data import PowerSpectraLoader
from lsskit.power import tools

class RunPB(PowerSpectraLoader):
    name = "RunPB"

    @classmethod
    def register(cls):
        PowerSpectraLoader.store_class(cls)

    def 