import argparse
import importlib
import numpy as np
import os
import sys
import torch
from src.model import SSD, ResNet
import cognata_labels

def main(args):
    config = importlib.import_module('config.' + args.config)
    image_size = config.model['image_size']
    num_classes=len(cognata_labels.label_map.keys())
    model = SSD(config.model, backbone=ResNet(config.model), num_classes=num_classes)
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    model.load_state_dict(checkpoint["model_state_dict"])
    input_image = torch.rand(1, 3, image_size[0], image_size[1])
    torch.onnx.export(model, input_image, args.saved_onnx_path, 
                      export_params=True, opset_version=11, do_constant_folding=True, 
                      input_names=['input_image'],
                      #dynamic_axes={'input_image': [0, 2, 3]},
                      output_names=['ploc', 'plabel'])
    print('finished')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Configuration Parameters')
    parser.add_argument("--checkpoint", type=str)
    parser.add_argument("--config", type=str)
    parser.add_argument('--saved_onnx_path', default='ssd_resnet50.onnx',
                        help='your saved onnx path')
    args = parser.parse_args()

    main(args)