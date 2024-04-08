"""
CM interface for ABTF model to plug to MLPerf loadgen

Developer: Grigori Fursin
"""


import importlib

from src.model import SSD, ResNet


def model_init(checkpoint, key):
    import os
    import sys

    cur_path=os.path.dirname(os.path.abspath(__file__))
 
    sys.path.insert(0, cur_path)
    config = importlib.import_module('config.baseline_8MP')
    del(sys.path[0])

    model = SSD(config.model, backbone=ResNet(config.model), num_classes=16)

    model.load_state_dict(checkpoint[key])

    return model
