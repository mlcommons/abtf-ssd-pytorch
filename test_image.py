"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""

import numpy as np
import argparse
import importlib
import torch
from src.transform import SSDTransformer
import cv2
from PIL import Image

from src.utils import generate_dboxes, Encoder, colors, coco_classes
from src.model import SSD, ResNet


def get_args():
    parser = argparse.ArgumentParser("Implementation of SSD")
    parser.add_argument("--input", type=str, required=True, help="the path to input image")
    parser.add_argument("--cls-threshold", type=float, default=0.3)
    parser.add_argument("--nms-threshold", type=float, default=0.5)
    parser.add_argument("--pretrained-model", type=str, default="trained_models/SSD.pth")
    parser.add_argument("--output", type=str, default=None, help="the path to output image")
    parser.add_argument("--dataset", default='Cognata', type=str)
    parser.add_argument("--config", default='config', type=str)
    args = parser.parse_args()
    return args


def test(opt):
    import os
    device = os.environ.get('CM_DEVICE','')
    if device == 'cuda' and not torch.cuda.is_available():
        print ('')
        print ('Error: CUDA is forced but not available...')
        exit(1)

    to_export_model = os.environ.get('CM_ABTF_EXPORT_MODEL_TO_ONNX','')
    exported = False

    config = importlib.import_module('config.' + opt.config)
    image_size = config.model['image_size']

#    model = SSD(backbone=ResNet())
    model = SSD(config.model, backbone=ResNet(config.model), num_classes=16) #num_classes)
#    checkpoint = torch.load(opt.pretrained_model)
    checkpoint = torch.load(opt.pretrained_model, map_location=torch.device(device))
    model.load_state_dict(checkpoint["model_state_dict"])
    if device=='cuda':
        model.cuda()
    model.eval()
#    dboxes = generate_dboxes()
    dboxes = generate_dboxes(config.model, model="ssd")

    transformer = SSDTransformer(dboxes, image_size, val=True)
    img = Image.open(opt.input).convert("RGB")
    img, _, _, _ = transformer(img, None, torch.zeros(1,4), torch.zeros(1))
    encoder = Encoder(dboxes)

    if torch.cuda.is_available():
        img = img.cuda()
    with torch.no_grad():
        inp = img.unsqueeze(dim=0)
        ploc, plabel = model(inp)
        result = encoder.decode_batch(ploc, plabel, opt.nms_threshold, 20)[0]

        if to_export_model!='' and not exported:
            torch.onnx.export(model,
                 inp,
                 to_export_model,
                 verbose=True,
                 input_names=['input'],
                 output_names=['output'],
                 export_params=True,
                 )

            exported = True     
        
        
        loc, label, prob = [r.cpu().numpy() for r in result]
        best = np.argwhere(prob > opt.cls_threshold).squeeze(axis=1)
        loc = loc[best]
        label = label[best]
        prob = prob[best]
        output_img = cv2.imread(opt.input)
        if len(loc) > 0:
            height, width, _ = output_img.shape
            loc[:, 0::2] *= width
            loc[:, 1::2] *= height
            loc = loc.astype(np.int32)
            for box, lb, pr in zip(loc, label, prob):
#                category = test_set.label_info[lb]
                category = coco_classes[lb]
                color = colors[lb]
                xmin, ymin, xmax, ymax = box
                cv2.rectangle(output_img, (xmin, ymin), (xmax, ymax), color, 2)
                text_size = cv2.getTextSize(category + " : %.2f" % pr, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
                cv2.rectangle(output_img, (xmin, ymin), (xmin + text_size[0] + 3, ymin + text_size[1] + 4), color,
                              -1)
                cv2.putText(
                    output_img, category + " : %.2f" % pr,
                    (xmin, ymin + text_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1,
                    (255, 255, 255), 1)
        if opt.output is None:
            output = "{}_prediction.jpg".format(opt.input[:-4])
        else:
            output = opt.output
        cv2.imwrite(output, output_img)


if __name__ == "__main__":
    opt = get_args()
    test(opt)
