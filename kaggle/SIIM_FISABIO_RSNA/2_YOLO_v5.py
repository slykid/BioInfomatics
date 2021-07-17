import os
import gc
import cv2
import yaml
from tqdm import tqdm
from shutil import copyfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
import torch
import tensorflow as tf

# 0. Hyperparameter
TRAIN_PATH = "data/SIIM_FISABIO_RSNA/prep/resized-to-256px-jpg/train/"
IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 10

class_label_to_id = {
    'Negative for Pneumonia': 0,
    'Typical Appearance': 1,
    'Indeterminate Appearance': 2,
    'Atypical Appearance': 3
}

class_id_to_label = {val: key for key, val in class_label_to_id.items()}

# 1. Data load
train = pd.read_csv("data/SIIM_FISABIO_RSNA/prep/train.csv")

# 2. Data split
train_df, valid_df = train_test_split(train, test_size=0.2, random_state=42, stratify=train.image_level.values)

train_df.loc[:, 'split'] = 'train'
valid_df.loc[:, 'split'] = 'valid'

df = pd.concat([train_df, valid_df]).reset_index(drop=True)
print(f'Size of dataset: {len(df)}, training images: {len(train_df)}. validation images: {len(valid_df)}')

# 3. make image
os.makedirs('tmp/covid/images/train', exist_ok=True)
os.makedirs('tmp/covid/images/valid', exist_ok=True)

os.makedirs('tmp/covid/labels/train', exist_ok=True)
os.makedirs('tmp/covid/labels/valid', exist_ok=True)

for i in tqdm(range(len(df))):
    row = df.loc[i]
    if row.split == 'train':
        copyfile(row.path, f'tmp/covid/images/train/{row.id}.jpg')
    else:
        copyfile(row.path, f'tmp/covid/images/valid/{row.id}.jpg')

# 4. Train parameter setting
data_yaml = dict(
    train = '../covid/images/train',
    val = '../covid/images/valid',
    nc = 2,
    names = ['none', 'opacity']
)

# Note that I am creating the file in the yolov5/data/ directory.
with open('tmp/yolov5/data/data.yaml', 'w') as outfile:
    yaml.dump(data_yaml, outfile, default_flow_style=True)

def get_bbox(row):
    bboxes = []
    bbox = []
    for i, l in enumerate(row.label.split(' ')):
        if (i % 6 == 0) | (i % 6 == 1):
            continue
        bbox.append(float(l))
        if i % 6 == 5:
            bboxes.append(bbox)
            bbox = []

    return bboxes

# Scale the bounding boxes according to the size of the resized image.
def scale_bbox(row, bboxes):
    # Get scaling factor
    scale_x = IMG_SIZE / row.dim1
    scale_y = IMG_SIZE / row.dim0

    scaled_bboxes = []
    for bbox in bboxes:
        x = int(np.round(bbox[0] * scale_x, 4))
        y = int(np.round(bbox[1] * scale_y, 4))
        x1 = int(np.round(bbox[2] * (scale_x), 4))
        y1 = int(np.round(bbox[3] * scale_y, 4))

        scaled_bboxes.append([x, y, x1, y1])  # xmin, ymin, xmax, ymax

    return scaled_bboxes

# Convert the bounding boxes in YOLO format.
def get_yolo_format_bbox(img_w, img_h, bboxes):
    yolo_boxes = []
    for bbox in bboxes:
        w = bbox[2] - bbox[0]  # xmax - xmin
        h = bbox[3] - bbox[1]  # ymax - ymin
        xc = bbox[0] + int(np.round(w / 2))  # xmin + width/2
        yc = bbox[1] + int(np.round(h / 2))  # ymin + height/2

        yolo_boxes.append([xc / img_w, yc / img_h, w / img_w, h / img_h])  # x_center y_center width height

    return yolo_boxes

# 5. train image setting
for i in tqdm(range(len(df))):
    row = df.loc[i]
    # Get image id
    img_id = row.id
    # Get split
    split = row.split
    # Get image-level label
    label = row.image_level

    if row.split == 'train':
        file_name = f'tmp/covid/labels/train/{row.id}.txt'
    else:
        file_name = f'tmp/covid/labels/valid/{row.id}.txt'

    if label == 'opacity':
        # Get bboxes
        bboxes = get_bbox(row)
        # Scale bounding boxes
        scale_bboxes = scale_bbox(row, bboxes)
        # Format for YOLOv5
        yolo_bboxes = get_yolo_format_bbox(IMG_SIZE, IMG_SIZE, scale_bboxes)

        with open(file_name, 'w') as f:
            for bbox in yolo_bboxes:
                bbox = [1] + bbox
                bbox = [str(i) for i in bbox]
                bbox = ' '.join(bbox)
                f.write(bbox)
                f.write('\n')

# 6. Train model
# 학습 명령어
# python train.py --img 256 \
#                  --batch 16 \
#                  --epochs 10 \
#                  --data data.yaml \
#                  --weights yolov5s.pt \
#                  --save_period 1\
#                  --project kaggle-siim-covid

# 7. inference
TEST_PATH = "data/SIIM_FISABIO_RSNA/prep/resized-to-256px-jpg/test/"
MODEL_PATH = "data/SIIM_FISABIO_RSNA/tmp/yolov5/kaggle-siim-covid/exp/weights/best.pt"

# python detect.py --weights "kaggle-siim-covid/exp2/weights/best.pt" \
#                   --source "D:/workspace/BioInfomatics/data/SIIM_FISABIO_RSNA/prep/resized-to-256px-jpg/test/" \
#                   --img 256 \
#                   --conf 0.281 \
#                   --iou-thres 0.5 \
#                   --max-det 3 \
#                   --save-txt \
#                   --save-conf

# 8. Make Submission
PRED_PATH = 'tmp/yolov5/runs/detect/exp/labels'
prediction_files = os.listdir(PRED_PATH)
print('Number of test images predicted as opaque: ', len(prediction_files))

# The submisison requires xmin, ymin, xmax, ymax format.
# YOLOv5 returns x_center, y_center, width, height
def correct_bbox_format(bboxes):
    correct_bboxes = []
    for b in bboxes:
        xc, yc = int(np.round(b[0] * IMG_SIZE)), int(np.round(b[1] * IMG_SIZE))
        w, h = int(np.round(b[2] * IMG_SIZE)), int(np.round(b[3] * IMG_SIZE))

        xmin = xc - int(np.round(w / 2))
        xmax = xc + int(np.round(w / 2))
        ymin = yc - int(np.round(h / 2))
        ymax = yc + int(np.round(h / 2))

        correct_bboxes.append([xmin, xmax, ymin, ymax])

    return correct_bboxes

# Read the txt file generated by YOLOv5 during inference and extract
# confidence and bounding box coordinates.
def get_conf_bboxes(file_path):
    confidence = []
    bboxes = []
    with open(file_path, 'r') as file:
        for line in file:
            preds = line.strip('\n').split(' ')
            preds = list(map(float, preds))
            confidence.append(preds[-1])
            bboxes.append(preds[1:-1])
    return confidence, bboxes

# Prediction loop for submission
sub_df = pd.read_csv('data/SIIM_FISABIO_RSNA/sample_submission.csv')
sub_df.tail()

predictions = []

for i in tqdm(range(len(sub_df))):
    row = sub_df.loc[i]
    id_name = row.id.split('_')[0]
    id_level = row.id.split('_')[-1]

    if id_level == 'study':
        # do study-level classification
        predictions.append("Negative 1 0 0 1 1")  # dummy prediction

    elif id_level == 'image':
        # we can do image-level classification here.
        # also we can rely on the object detector's classification head.
        # for this example submisison we will use YOLO's classification head.
        # since we already ran the inference we know which test images belong to opacity.
        if f'{id_name}.txt' in prediction_files:
            # opacity label
            confidence, bboxes = get_conf_bboxes(f'{PRED_PATH}/{id_name}.txt')
            bboxes = correct_bbox_format(bboxes)
            pred_string = ''
            for j, conf in enumerate(confidence):
                pred_string += f'opacity {conf} ' + ' '.join(map(str, bboxes[j])) + ' '
            predictions.append(pred_string[:-1])
        else:
            predictions.append("None 1 0 0 1 1")

sub_df['PredictionString'] = predictions
sub_df.to_csv('data/SIIM_FISABIO_RSNA/submission.csv', index=False)
sub_df.tail()