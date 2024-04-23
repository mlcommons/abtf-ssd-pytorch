"""
Collective Mind Connector for the ABTF model 
to plug it to the CM workflow with MLPerf loadgen

Developer: Grigori Fursin
"""


import importlib

from src.model import SSD, ResNet


def model_init(checkpoint, cfg):
    import os
    import sys

    checkpoint_key = cfg.get('checkpoint_key', 'model_state_dict')
    model_config = cfg.get('config', 'baseline_8MP')
    num_classes = int(cfg.get('num_classes', 16))

    cur_path=os.path.dirname(os.path.abspath(__file__))
 
    sys.path.insert(0, cur_path)
    config = importlib.import_module('config.'+model_config)
    del(sys.path[0])

    model = SSD(config.model, backbone=ResNet(config.model), num_classes = num_classes)

    model.load_state_dict(checkpoint[checkpoint_key])

    return {'return':0, 'model':model}
