import os
import shutil
import importlib
from argparse import ArgumentParser
from torchinfo import summary

from src.model import SSD, SSDLite, ResNet, MobileNetV2
from src.utils import generate_dboxes, Encoder, coco_classes

def get_args():
    parser = ArgumentParser(description="model summary given config file")
    parser.add_argument("--config", default='config', type=str)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--batch-size", default=4, type=int)
    args = parser.parse_args()
    return args

def main(opt):
    config = importlib.import_module('config.' + opt.config)
    model = SSD(backbone=ResNet(), num_classes=opt.num_classes)
    image_size = config.model['image_size']
    summary(model, input_size=(opt.batch_size, 3, image_size[0], image_size[1]))

if __name__ == "__main__":
    opt = get_args()
    main(opt)