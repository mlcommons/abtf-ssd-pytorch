import os
import csv
import ast
from argparse import ArgumentParser
from tqdm import tqdm

def scan_cognata(root, folders, cameras):
    files = []
    for folder in folders:
        for camera in cameras:
            ann_folder = os.path.join(root, folder, camera + '_ann')
            img_folder = os.path.join(root, folder, camera + '_png')
            ann_files = sorted([os.path.join(ann_folder, f) for f in os.listdir(ann_folder) if os.path.isfile(os.path.join(ann_folder, f))])
            img_files = sorted([os.path.join(img_folder, f) for f in os.listdir(img_folder) if os.path.isfile(os.path.join(img_folder, f))])
            for i in range(len(ann_files)):
                files.append({'img': img_files[i], 'ann': ann_files[i]})
    
    return object_labels(files)

def object_labels(files):
    counter = 1
    label_map = {}
    label_info = {}
    object_count = {}
    label_info[0] = "background"
    label_map[0] = 0
    object_count[0] = 0
    for i in tqdm(range(len(files))):
        with open(files[i]['ann']) as f:
            reader = csv.reader(f)
            rows = list(reader)
            header = rows[0]
            annotations = rows[1:]
            class_index = header.index('object_class')
            class_name_index = header.index('object_class_name')
            bbox_index = header.index('bounding_box_2D')
            for annotation in annotations:
                label = ast.literal_eval(annotation[class_index])
                bbox = annotation[bbox_index]
                bbox = ast.literal_eval(bbox)
                object_area = (bbox[2]-bbox[0])*(bbox[3]-bbox[1])
                if object_area <= 20:
                    continue
                if label not in label_map:
                    label_map[label] = counter
                    object_count[label] = 0
                    label_info[counter] = annotation[class_name_index]
                    counter += 1
                object_count[label] += 1
    return label_map, label_info, object_count

def get_args():
    parser = ArgumentParser(description="scan objects in dataset given config file")
    parser.add_argument("--data-path", default='/cognata', type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    opt = get_args()
    folders = [
    '10001_Urban_Clear_Noon', '10006_Urban_Clear_Noon', '10011_Urban_Rain_Evening', '10016_Highway_Clear_Evening', '10021_Urban_Clear_Morning',
    '10026_Urban_HeavyRain_Afternoon', '10002_Urban_Clear_Morning', '10007_Highway_Clear_Morning', '10012_Urban_Rain_Evening', '10017_Urban_Rain_Evening', '10022_Urban_Clear_Morning',      
    '10003_Urban_Clear_Noon', '10008_Highway_Clear_Noon', '10013_Highway_Rain_Morning', '10018_Urban_Rain_Evening', '10023_Urban_Rain_Morning',
    '10004_Urban_Clear_Noon', '10009_Urban_Rain_Morning', '10014_Highway_Rain_Noon', '10019_Urban_HeavyRain_Noon', '10024_Urban_Rain_Evening',
    '10005_Urban_Clear_Noon', '10010_Urban_Rain_Morning', '10015_Highway_Clear_Evening', '10020_Urban_HeavyRain_Afternoon', '10025_Urban_HeavyRain_Evening'
    ]
    cameras = ['Cognata_Camera_01_8M', 'Cognata_Camera_02_8M', 'Cognata_Camera_03_8M']
    label_map, label_info, object_count = scan_cognata(opt.data_path, folders, cameras)
    for label in label_map:
        print(str(label_info[label_map[label]]) + ': ' + str(object_count[label]))
