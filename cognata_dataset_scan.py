import os
import csv
import ast
from argparse import ArgumentParser
from tqdm import tqdm
import math
import csv

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
    
    ignore_classes = [2, 25, 31]#pole, parking and fence. Refer to the cognata dataset pdfs for class references
    return object_labels(files, ignore_classes)

def round_down_int(x, y):
    return math.floor(x/y)*y

def object_labels(files, ignore_classes):
    counter = 1
    label_map = {}
    label_info = {}
    object_count = {}
    stats = {'height': {}, 'width': {}, 'area': {}, 'aspect_ratio': {}}
    max_height = 2400
    max_width = 4000
    max_area = 500000
    max_ratio = 10
    stats['height'] = dict.fromkeys(range(0, max_height+50, 50), 0)
    stats['width'] = dict.fromkeys(range(0, max_width+50, 50), 0)
    stats['area'] = dict.fromkeys(range(0, max_area+100, 1000), 0)
    stats['aspect_ratio'] = dict.fromkeys(range(1, max_ratio+1), 0)

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
                obj_height = bbox[3]-bbox[1]
                obj_width = bbox[2]-bbox[0]
                object_area = obj_height*obj_width
                if object_area <= 20 or int(label) in ignore_classes:
                    continue
                aspect_ratio = obj_height/obj_width
                if aspect_ratio < 1:
                    aspect_ratio = 1/aspect_ratio
                bin = round_down_int(object_area, 1000)
                if bin > max_area:
                    stats['area'][max_area] += 1
                else:
                    stats['area'][bin] += 1
                bin = round_down_int(obj_height, 50)
                if bin > max_height:
                    stats['height'][max_height] += 1
                else:
                    stats['height'][bin] += 1
                bin = round_down_int(obj_width, 50)
                if bin > max_width:
                    stats['width'][max_height] += 1
                else:
                    stats['width'][bin] += 1
                bin = int(math.floor(aspect_ratio))
                if bin > max_ratio:
                    stats['aspect_ratio'][max_ratio] += 1
                else:
                    stats['aspect_ratio'][bin] += 1
                if label not in label_map:
                    label_map[label] = counter
                    object_count[label] = 0
                    label_info[counter] = annotation[class_name_index]
                    counter += 1
                object_count[label] += 1
    return label_map, label_info, object_count, stats

def get_args():
    parser = ArgumentParser(description="scan objects in dataset")
    parser.add_argument("--data-path", default='/cognata', type=str)
    args = parser.parse_args()
    return args

def write_stat_csv(f_name, d):
    with open(f_name, 'w') as f:
        w = csv.DictWriter(f, d.keys())
        w.writeheader()
        w.writerow(d)

if __name__ == "__main__":
    opt = get_args()
    '''folders = [
    '10001_Urban_Clear_Noon', '10006_Urban_Clear_Noon', '10011_Urban_Rain_Evening', '10016_Highway_Clear_Evening', '10021_Urban_Clear_Morning',
    '10026_Urban_HeavyRain_Afternoon', '10002_Urban_Clear_Morning', '10007_Highway_Clear_Morning', '10012_Urban_Rain_Evening', '10017_Urban_Rain_Evening', '10022_Urban_Clear_Morning',      
    '10003_Urban_Clear_Noon', '10008_Highway_Clear_Noon', '10013_Highway_Rain_Morning', '10018_Urban_Rain_Evening', '10023_Urban_Rain_Morning',
    '10004_Urban_Clear_Noon', '10009_Urban_Rain_Morning', '10014_Highway_Rain_Noon', '10019_Urban_HeavyRain_Noon', '10024_Urban_Rain_Evening',
    '10005_Urban_Clear_Noon', '10010_Urban_Rain_Morning', '10015_Highway_Clear_Evening', '10020_Urban_HeavyRain_Afternoon', '10025_Urban_HeavyRain_Evening'
    ]'''
    folders = ['10001_Urban_Clear_Noon']
    #cameras = ['Cognata_Camera_01_8M', 'Cognata_Camera_02_8M', 'Cognata_Camera_03_8M']
    cameras = ['Cognata_Camera_01_8M']
    label_map, label_info, object_count, stats = scan_cognata(opt.data_path, folders, cameras)
    for label in label_map:
        print(str(label_info[label_map[label]]) + ': ' + str(object_count[label]))
    
    write_stat_csv('height.csv', stats['height'])
    write_stat_csv('width.csv', stats['width'])
    write_stat_csv('area.csv', stats['area'])
    write_stat_csv('aspect_ratio.csv', stats['aspect_ratio'])
