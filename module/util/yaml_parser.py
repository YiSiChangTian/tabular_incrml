import os
from yaml import load

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader


class YamlParser(object):

    @staticmethod
    def parse(path, mode = None):
        if os.path.exists(path):
            with open(path, "rt") as fh:
                config = load(fh, Loader=Loader)
                if mode is not None:
                    config = config[mode]
        else:
            raise ValueError("CONFIG FILE NOT FOUND")
            os._exit(1)
        return config
