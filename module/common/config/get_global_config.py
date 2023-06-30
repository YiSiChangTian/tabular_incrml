
__all__ = ['global_config']
import os
from module.util.yaml_parser import YamlParser
conf_path = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))),
    "conf", 
    'global.yaml')
global_config = YamlParser.parse(path=conf_path)