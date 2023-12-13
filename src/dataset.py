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
    def __init__(self, root, folders, transform=None):
        ann_files = []
        img_files = []
        self.label_map = {}
        self.label_info = {}
        for folder in folders:
            ann_folder = os.path.join(root, folder + '_ann')
            img_folder = os.path.join(root, folder + '_png')
            ann_files += [os.path.join(ann_folder, f) for f in os.listdir(ann_folder) if os.path.isfile(os.path.join(ann_folder, f))]
            img_files += [os.path.join(img_folder, f) for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))]
        self.transform = transform
        self.root = root
        self.ann_files = ann_files
        self.img_files = img_files
        self.object_labels()
    
    def __len__(self):
        return len(self.img_files)
    
    def __getitem__(self, idx):
        img = Image.open(self.img_files[idx]).convert('RGB')
        width, height = img.size
        boxes = []
        labels = []
        with open(self.ann_files[idx]) as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            annotations = rows[1:]
            bbox_index = header.index('bounding_box_2D')
            class_index = header.index('object_class')
            for annotation in annotations:
                bbox = annotation[bbox_index]
                bbox = ast.literal_eval(bbox)
                bbox_width = bbox[2] - bbox[0]
                bbox_height = bbox[3] - bbox[1]
                coco_format = [bbox[0], bbox[1], bbox_width, bbox_height]
                boxes.append([coco_format[0] / width, coco_format[1] / height, (coco_format[0] + coco_format[2]) / width, (coco_format[1] + coco_format[3]) / height])
                label = ast.literal_eval(annotation[class_index])
                labels.append(label)
            
            boxes = torch.tensor(boxes)
            labels = torch.tensor(labels)
        if self.transform is not None:
            image, (height, width), boxes, labels = self.transform(img, (height, width), boxes, labels)
        return image, idx, (height, width), boxes, labels

    def object_labels(self):
        for ann_file in self.ann_files:
            with open(ann_file) as f:
                reader = csv.reader(f)
                rows = list(reader)
                header = rows[0]
                annotations = rows[1:]
                class_index = header.index('object_class')
                class_name_index = header.index('object_class_name')
                counter = 1
                for annotation in annotations:
                    label = ast.literal_eval(annotation[class_index])
                    if label not in self.label_map:
                        self.label_map[label] = counter
                        self.label_info[counter] = annotation[class_name_index]
                        counter += 1


