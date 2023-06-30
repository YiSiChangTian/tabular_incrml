import os

from module.util.yaml_parser import YamlParser

from logging import Logger

class TaskConfig(object):
    __species = None
    __first_init = True

    def __new__(cls, *args, **kwargs):
        if cls.__species is None:
            cls.__species = object.__new__(cls)
        return cls.__species

    def __init__(self, conf_file = None):
        if self.__first_init:
            self.__class__.__first_init = False
            if conf_file is not None:
                self.cfg_path = os.path.abspath(conf_file)
                self.cfg = YamlParser.parse(conf_file)
            else:
                Logger.error(msg="Task config path missing")
