"""
@author: Viet Nguyen (nhviet1009@gmail.com)
"""
import torch
from torchvision.datasets import CocoDetection
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import os
from PIL import Image
import csv
import ast
import random

def collate_fn(batch):
    items = list(zip(*batch))
    items[0] = default_collate([i for i in items[0] if torch.is_tensor(i)])
    items[1] = list([i for i in items[1] if i])
    items[2] = list([i for i in items[2] if i])
    items[3] = default_collate([i for i in items[3] if torch.is_tensor(i)])
    items[4] = default_collate([i for i in items[4] if torch.is_tensor(i)])
    return items


class CocoDataset(CocoDetection):
    def __init__(self, root, year, mode, transform=None):
        annFile = os.path.join(root, "annotations", "instances_{}{}.json".format(mode, year))
        root = os.path.join(root, "{}{}".format(mode, year))
        super(CocoDataset, self).__init__(root, annFile)
        self._load_categories()
        self.transform = transform

    def _load_categories(self):

        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x["id"])

        self.label_map = {}
        self.label_info = {}
        counter = 1
        self.label_info[0] = "background"
        for c in categories:
            self.label_map[c["id"]] = counter
            self.label_info[counter] = c["name"]
            counter += 1

    def __getitem__(self, item):
        image, target = super(CocoDataset, self).__getitem__(item)
        width, height = image.size
        boxes = []
        labels = []
        if len(target) == 0:
            return None, None, None, None, None
        for annotation in target:
            bbox = annotation.get("bbox")
            boxes.append([bbox[0] / width, bbox[1] / height, (bbox[0] + bbox[2]) / width, (bbox[1] + bbox[3]) / height])
            labels.append(self.label_map[annotation.get("category_id")])
        boxes = torch.tensor(boxes)
        labels = torch.tensor(labels)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(image, (height, width), boxes, labels)
        return image, target[0]["image_id"], (height, width), boxes, labels

class Cognata(Dataset):
    def __init__(self, label_map, label_info, files, transform=None):
        self.label_map = label_map
        self.label_info = label_info
        self.transform = transform
        self.files = files
        self.ignore_classes = [2, 25, 31]
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img = Image.open(self.files[idx]['img']).convert('RGB')
        width, height = img.size
        boxes = []
        labels = []
        gt_boxes = []
        with open(self.files[idx]['ann']) as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            annotations = rows[1:]
            bbox_index = header.index('bounding_box_2D')
            class_index = header.index('object_class')
            for annotation in annotations:
                bbox = annotation[bbox_index]
                bbox = ast.literal_eval(bbox)
                object_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                label = ast.literal_eval(annotation[class_index])
                if object_area <= 20 and not int(label) in self.ignore_classes:
                    continue
                boxes.append([bbox[0] / width, bbox[1] / height, bbox[2] / width, bbox[3] / height])
                label = self.label_map[label]
                gt_boxes.append([bbox[0], bbox[1], bbox[2], bbox[3], label, 0, 0])
                labels.append(label)
            
            boxes = torch.tensor(boxes)
            labels = torch.tensor(labels)
            gt_boxes = torch.tensor(gt_boxes)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(img, (height, width), boxes, labels, max_num=500)
        return image, idx, (height, width), boxes, labels, gt_boxes

def object_labels(files, ignore_classes):
    counter = 1
    label_map = {}
    label_info = {}
    label_info[0] = "background"
    label_map[0] = 0
    for file in files:
        with open(file['ann']) as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            annotations = rows[1:]
            class_index = header.index('object_class')
            class_name_index = header.index('object_class_name')
            for annotation in annotations:
                label = ast.literal_eval(annotation[class_index])
                if label not in label_map and not int(label) in ignore_classes:
                    label_map[label] = counter
                    label_info[counter] = annotation[class_name_index]
                    counter += 1
    return label_map, label_info

def prepare_cognata(root, folders, cameras):
    files = []
    for folder in folders:
        for camera in cameras:
            ann_folder = os.path.join(root, folder, camera + '_ann')
            img_folder = os.path.join(root, folder, camera + '_png')
            ann_files = sorted([os.path.join(ann_folder, f) for f in os.listdir(ann_folder) if os.path.isfile(os.path.join(ann_folder, f))])
            img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))])
            for i in range(len(ann_files)):
                with open(ann_files[i]) as f:
                    reader = csv.reader(f)
                    rows = list(reader)
                    header = rows[0]
                    annotations = rows[1:]
                    bbox_index = header.index('bounding_box_2D')
                    for annotation in annotations:
                        bbox = annotation[bbox_index]
                        bbox = ast.literal_eval(bbox)
                        object_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                        if object_area > 20:
                            files.append({'img': img_files[i], 'ann': ann_files[i]})
                            break
    
    ignore_classes = [2, 25, 31]
    label_map, label_info = object_labels(files, ignore_classes)
    return files, label_map, label_info

def train_val_split(files):
    random.Random(5).shuffle(files)
    val_index = round(len(files)*0.8)
    return {'train': files[:val_index], 'val': files[val_index:]}
