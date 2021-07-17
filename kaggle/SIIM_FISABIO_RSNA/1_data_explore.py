# train_study_level.csv
# - the train study-level metadata, with one row for each study, including correct labels.
#
# id - unique study identifier
# Negative for Pneumonia - 1 if the study is negative for pneumonia, 0 otherwise
# Typical Appearance - 1 if the study has this appearance, 0 otherwise
# Indeterminate Appearance  - 1 if the study has this appearance, 0 otherwise
# Atypical Appearance  - 1 if the study has this appearance, 0 otherwise

# train_image_level.csv
# - the train image-level metadata, with one row for each image, including both correct labels and any bounding boxes in a dictionary format. Some images in both test and train have multiple bounding boxes.
#
# id - unique image identifier
# boxes - bounding boxes in easily-readable dictionary format
# label - the correct prediction label for the provided bounding boxes

import os
import re
import gc
import cv2
import ast
import cv2
from tqdm import tqdm
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from sklearn.model_selection import StratifiedKFold
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

import wandb # 0.10.33
from wandb.keras import WandbCallback

wandb.login()

# GPU 확인
gpu = tf.config.list_physical_devices('GPU')
if gpu:
    try:
        for unit in gpu:
            tf.config.experimental.set_memory_growth(unit, True)
        logical_gpu = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpu), "Physical GPU, ", len(logical_gpu), "Logical GPU")
    except RuntimeError as e:
        print(e)

# 0. Hyperpameters
TRAIN_PATH = "data/SIIM_FISABIO_RSNA/prep/resized-to-256px-jpg/train/"
IMG_SIZE = 256
NUM_SAMPLES_TO_VIZ = 32

# 1. 데이터 확인
train_image = pd.read_csv("data/SIIM_FISABIO_RSNA/train_image_level.csv")  # 6,054 / image
train_study = pd.read_csv("data/SIIM_FISABIO_RSNA/train_study_level.csv")  # 6,334 / label

print(train_image.columns)  # 'id', 'boxes', 'label', 'StudyInstanceUID'
print(train_image.info())
print(train_study.columns)  # 'id', 'Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance'
print(train_study.info())

# image 데이터 정제
train_image["id"] = train_image.apply(lambda row: row.id.split('_')[0], axis=1)
train_image["path"] = train_image.apply(lambda row: TRAIN_PATH + row.id + ".jpg", axis=1)  # 이미지 경로 저장
train_image["image_level"] = train_image.apply(lambda row: row.label.split(' ')[0], axis=1)

# study 데이터 정제
train_study["id"] = train_study.apply(lambda row: row.id.split('_')[0], axis=1)
train_study.columns = ['StudyInstanceUID', 'Negative for Pneumonia', 'Typical Appearance', 'Indeterminate Appearance', 'Atypical Appearance']

# train 데이터 결합
train = train_image.merge(train_study, on='StudyInstanceUID', how="left")
print(train.columns)
# 'id', 'boxes', 'label', 'StudyInstanceUID', 'path', 'image_level',
# 'Negative for Pneumonia', 'Typical Appearance',
# 'Indeterminate Appearance', 'Atypical Appearance'

print(train.loc[0])

# 학습 데이터 수 확인
print(f"Number of unique image in training dataset: {len(train)}")

# 박스가 없는 데이터 수
bbox_nan_num = train['boxes'].isna().sum()
print(f"Number of images without any bbox annotation: {bbox_nan_num}")

labels = train[["Negative for Pneumonia","Typical Appearance","Indeterminate Appearance","Atypical Appearance"]].values
labels = np.argmax(labels, axis=1)

train['study_level'] = labels
print(train.loc[0])

class_label_to_id = {
    'Negative for Pneumonia': 0,
    'Typical Appearance': 1,
    'Indeterminate Appearance': 2,
    'Atypical Appearance': 3
}

class_id_to_label = {val: key for key, val in class_label_to_id.items()}

# 각 id 에 대한 이미지 정보
meta_df = pd.read_csv("data/SIIM_FISABIO_RSNA/meta.csv")
train_meta_df = meta_df.loc[meta_df.split == "train"]
train_meta_df.columns = ["id", "dim0", "dim1", "split"]

train = train.merge(train_meta_df, on="id", how="left")

train.to_csv("data/SIIM_FISABIO_RSNA/prep/train.csv", index=False)

# 박스 치기
opacity_df = train.dropna(subset = ["boxes"], inplace=False)
opacity_df = opacity_df.reset_index(drop=True)

# 박스 생성
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

# 스케일링 박스
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

# interactive image
def wandb_bbox(image, bboxes, true_label, class_id_to_label):
    all_boxes = []
    for bbox in bboxes:
        box_data = {"position": {
            "minX": bbox[0],
            "minY": bbox[1],
            "maxX": bbox[2],
            "maxY": bbox[3]
        },
            "class_id": int(true_label),
            "box_caption": class_id_to_label[true_label],
            "domain": "pixel"}
        all_boxes.append(box_data)

    return wandb.Image(image, boxes={
        "ground_truth": {
            "box_data": all_boxes,
            "class_labels": class_id_to_label
        }
    })

sampled_df = opacity_df.sample(NUM_SAMPLES_TO_VIZ).reset_index(drop=True)
run = wandb.init(project='kaggle-covid',
                 config={'competition': 'siim-fisabio-rsna', '_wandb_kernel': 'slykid'},
                 job_type='visualize_sample_bbox')

wandb_bbox_list = []
for i in tqdm(range(NUM_SAMPLES_TO_VIZ)):
    row = sampled_df.loc[i]
    # Load image
    image = cv2.imread(row.path)

    # Get bboxes
    bboxes = get_bbox(row)

    # Scale bounding boxes
    scale_bboxes = scale_bbox(row, bboxes)

    # Get ground truth label
    true_label = row.study_level

    wandb_bbox_list.append(wandb_bbox(image,
                                      scale_bboxes,
                                      true_label,
                                      class_id_to_label))

wandb.log({"radiograph": wandb_bbox_list})
run.finish()
