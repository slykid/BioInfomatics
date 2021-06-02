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
# ->


import re
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

import tensorflow as tf

train_study = pd.read_csv("data/SIIM_FISABIO_RSNA/train_study_level.csv")
train_image = pd.read_csv("data/SIIM_FISABIO_RSNA/train_image_level.csv")

print(train_study.columns)
print(train_study.info())

# study id, image id 컬럼 값 정제
train_study["uid"] = ""
for i in range(0, len(train_study["id"])):
    train_study["uid"][i] = re.sub("_study", "", str(train_study["id"][i]))

train_image["image_id"] = ""
for i in range(0, len(train_image["id"])):
    train_image["image_id"][i] = re.sub("_image", "", str(train_image["id"][i]))

print(train_study["uid"])
print(train_image["image_id"])

# train 데이터 결합
# left="id_1", right="StudyInstanceUID"
train = pd.merge(train_study, train_image, left_on="uid", right_on="StudyInstanceUID")
print(train.columns)

train.columns = ["study_id", "negative_pneumonia", "typical_appearance", "indeterminate_appearance", "atypical_appearance", "uid", "id_y", "boxes", "label", "study_uid", "image_id"]
print(train.columns)

train = train[["uid", "image_id", "negative_pneumonia", "typical_appearance", "indeterminate_appearance", "atypical_appearance", "boxes", "label"]]
print(train.columns)

# boxes, label 컬럼 정제 필요
# 1) label 컬럼 정제
print(train["label"][0])

train.to_csv("data/SIIM_FISABIO_RSNA/prep/train.csv", index=False)
