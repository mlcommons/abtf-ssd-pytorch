"""
Modifications by MLCommons from SSD-Pytorch (https://github.com/uvipen/SSD-pytorch) author: Viet Nguyen (nhviet1009@gmail.com)
Copyright 2024 MLCommons Association and Contributors

MIT License

Copyright (c) 2021 Viet Nguyen

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import os
import numpy as np
import argparse
import importlib
import torch
from src.dataset import CocoDataset, Cognata, prepare_cognata, train_val_split
from src.transform import SSDTransformer
import cv2
import shutil
import cognata_labels

from src.utils import generate_dboxes, Encoder, colors, coco_classes
from src.model import SSD, ResNet


def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--data-path", type=str, default="/cognata", help="the root folder of dataset")
    parser.add_argument("--cls-threshold", type=float, default=0.5)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default="predictions")
    parser.add_argument("--dataset", default='Cognata', type=str)
    parser.add_argument("--config", default='config', type=str)
    parser.add_argument("--box-thickness", type=int, default=2)
    parser.add_argument("--text-size", type=int, default=2)
    parser.add_argument("--max-boxes", type=int, default=100)
    args = parser.parse_args()
    return args


def test(opt):
    config = importlib.import_module('config.' + opt.config)
    image_size = config.model['image_size']
    dboxes = generate_dboxes(config.model, model="ssd")
    if opt.dataset == 'Cognata':
        folders = config.dataset['folders']
        cameras = config.dataset['cameras']
        ignore_classes = [2, 25, 31]
        files, label_map, label_info = prepare_cognata(opt.data_path, folders, cameras)
        #files = train_val_split(files)
        if 'use_label_info' in config.dataset and config.dataset['use_label_info']:
            label_map = cognata_labels.label_map
            label_info = cognata_labels.label_info
        test_set = Cognata(label_map, label_info, files, ignore_classes, SSDTransformer(dboxes, image_size, val=True))
        num_classes = len(label_map.keys())
        print(label_map)
        print(label_info)
    else:
        test_set = CocoDataset(opt.data_path, 2017, "val", SSDTransformer(dboxes, image_size, val=False))
        num_classes = len(coco_classes)
    encoder = Encoder(dboxes)
    model = SSD(config.model, backbone=ResNet(config.model), num_classes=num_classes)
    checkpoint = torch.load(opt.pretrained_model)
    model.load_state_dict(checkpoint["model_state_dict"])
    if torch.cuda.is_available():
        model.cuda()
    model.eval()
    if os.path.isdir(opt.output):
        shutil.rmtree(opt.output)
    os.makedirs(opt.output)

    for img, img_id, img_size, _, _, *other in test_set:
        if img is None:
            continue
        if torch.cuda.is_available():
            img = img.cuda()
        with torch.no_grad():
            ploc, plabel = model(img.unsqueeze(dim=0))
            result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, opt.max_boxes)[0]
            loc, label, prob = [r.cpu().numpy() for r in result]
            best = np.argwhere(prob > opt.cls_threshold).squeeze(axis=1)
            loc = loc[best]
            label = label[best]
            prob = prob[best]
            if len(loc) > 0:
                if opt.dataset == 'Cognata':
                    path = files[img_id]['img']
                    output_img = cv2.imread(path)
                    output_path = os.path.basename(path)[:-4]
                else:
                    path = test_set.coco.loadImgs(img_id)[0]["file_name"]
                    output_img = cv2.imread(os.path.join(opt.data_path, "val2017", path))
                    output_path = path[:-4]
                height, width, _ = output_img.shape
                loc[:, 0::2] *= width
                loc[:, 1::2] *= height
                loc = loc.astype(np.int32)
                for box, lb, pr in zip(loc, label, prob):
                    category = test_set.label_info[lb]
                    color = colors[lb]
                    xmin, ymin, xmax, ymax = box
                    cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, int(opt.box_thickness))
                    text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                    cv2.rectangle(output_img, (xmin, ymin-(text_size[1]*(opt.text_size + 1))), (xmin + text_size[0]*opt.text_size, ymin), color,
                                  -1)
                    cv2.putText(
                        output_img, category + " : %.2f" % pr,
                        (xmin, ymin), cv2.FONT_HERSHEY_PLAIN, opt.text_size,
                        (255, 255, 255), opt.text_size)
                    
                    cv2.imwrite("{}/{}_prediction.jpg".format(opt.output, output_path), output_img)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
