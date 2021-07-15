# 주의사항
## 아래의 코드는 어디까지나 샘플입니다.
## 때문에 모델의 정확도는 굉장히 낮으며, 사용자가 데이터 분석으로 통해 적절한 값을 찾고,
## 변환해서 사용하여 주시기 바랍니다.

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras as K

#  train dataset
train_patient = pd.read_csv("data/Cancer_Classification/train_data_clinical_patient.csv")

# test dataset
test_patient = pd.read_csv("data/Cancer_Classification/test_data_clinical_patient_v1.csv")

# patient 만으로 학습
x_train = train_patient[["AGE", "SEX", "WEIGHT", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC_STAGING_EDITION",
    "PATH_M_STAGE", "PATH_N_STAGE", "PATH_T_STAGE", "PERSON_NEOPLASM_CANCER_STATUS",
    "PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT", "RADIATION_THERAPY", "IN_PANCANPATHWAYS_FREEZE"]]

x_train.isnull().any()

# AGE 컬럼 수정
print(x_train.AGE.describe())
print(pd.unique(x_train.AGE))
x_train.AGE = x_train.AGE.fillna(np.round(np.mean(x_train.AGE))) # NaN 은 평균값으로 대처
x_train["AGE"] = x_train["AGE"].astype("int32")

# SEX 컬럼 수정
print(pd.unique(x_train["SEX"]))
x_train["SEX"].describe()
x_train["SEX"] = x_train["SEX"].fillna("Male") # NaN 은 최빈값으로 대처

for i in range(0, len(x_train["SEX"])):
        if x_train["SEX"][i] == "Male":
            x_train["SEX"][i] = 0
        elif x_train["SEX"][i] == "Female":
            x_train["SEX"][i] = 1
        else:
            x_train["SEX"][i] = 2

# WEIGHT
print(x_train.WEIGHT.describe())
print(pd.unique(x_train.WEIGHT))

x_train.WEIGHT = x_train.WEIGHT.fillna(np.mean(x_train.WEIGHT))

# AJCC_PATHOLOGIC_TUMOR_STAGE 컬럼 수정
pd.unique(x_train.AJCC_PATHOLOGIC_TUMOR_STAGE)
x_train.AJCC_PATHOLOGIC_TUMOR_STAGE = x_train.AJCC_PATHOLOGIC_TUMOR_STAGE.fillna("STAGE 0")

stage_level = ['STAGE 0', 'STAGE I', 'STAGE IA', 'STAGE IB', 'STAGE I/II (NOS)',
 'STAGE II', 'STAGE IIA', 'STAGE IIB', 'STAGE IIC',
 'STAGE III', 'STAGE IIIA', 'STAGE IIIB', 'STAGE IIIC',
 'STAGE IV', 'STAGE IVA', 'STAGE IVB']

for i in range(0, len(x_train)):
    if x_train["AJCC_PATHOLOGIC_TUMOR_STAGE"][i] in stage_level:
        x_train["AJCC_PATHOLOGIC_TUMOR_STAGE"][i] = stage_level.index(x_train["AJCC_PATHOLOGIC_TUMOR_STAGE"][i])

# AJCC_STAGING_EDITION 컬럼 수정
pd.unique(x_train["AJCC_STAGING_EDITION"])
x_train["AJCC_STAGING_EDITION"] = x_train["AJCC_STAGING_EDITION"].fillna("unknown")
stage_edition = ['1ST', '2ND', '3RD', '4TH', '5TH', '6TH', '7TH', 'unknown']
for i in range(0, len(x_train)):
    if x_train["AJCC_STAGING_EDITION"][i] in stage_edition:
        x_train["AJCC_STAGING_EDITION"][i] = stage_edition.index(x_train["AJCC_STAGING_EDITION"][i])

# PATH_STAGE 컬럼들 수정
print(pd.unique(x_train["PATH_M_STAGE"]))
print(pd.unique(x_train["PATH_N_STAGE"]))
print(pd.unique(x_train["PATH_T_STAGE"]))

x_train["PATH_M_STAGE"] = x_train["PATH_M_STAGE"].fillna("unknown")
x_train["PATH_N_STAGE"] = x_train["PATH_N_STAGE"].fillna("unknown")
x_train["PATH_T_STAGE"] = x_train["PATH_T_STAGE"].fillna("unknown")

path_m = ["M0", "M1", "M1A", "M1B", "M1C", "MX", "unknown"]
path_n = ["N0", "N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B", "N2C", "N3", "N3A", "N3B", "NX", "unknown"]
path_t = ["T0", "T1", "T1A", "T1B", "T2", "T2A", "T2B", "T3", "T3A", "T3B", "T4", "T4A", "T4B", "TX","TIS", "unknown"]

for i in range(0, len(x_train)):
    if x_train["PATH_M_STAGE"][i] in path_m:
        x_train["PATH_M_STAGE"][i] = path_m.index(x_train["PATH_M_STAGE"][i])

    if x_train["PATH_N_STAGE"][i] in path_n:
        x_train["PATH_N_STAGE"][i] = path_n.index(x_train["PATH_N_STAGE"][i])

    if x_train["PATH_T_STAGE"][i] in path_t:
        x_train["PATH_T_STAGE"][i] = path_t.index(x_train["PATH_T_STAGE"][i])

# PERSON_NEOPLASM_CANCER_STATUS 컬럼 수정
print(pd.unique(x_train.PERSON_NEOPLASM_CANCER_STATUS))
x_train["PERSON_NEOPLASM_CANCER_STATUS"] = x_train["PERSON_NEOPLASM_CANCER_STATUS"].fillna("Tumor Free")

for i in range(0, len(x_train)):
    if x_train["PERSON_NEOPLASM_CANCER_STATUS"][i].__eq__("Tumor Free"):
        x_train["PERSON_NEOPLASM_CANCER_STATUS"][i] = 0
    else :
        x_train["PERSON_NEOPLASM_CANCER_STATUS"][i] = 1

# PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT 컬럼 수정
pd.unique(x_train["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"])
x_train["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"] = x_train["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"].fillna("No")

for i in range(0, len(x_train)):
    if x_train["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"][i] == "Yes":
        x_train["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"][i] = 0
    else :
        x_train["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"][i] = 1

# RADIATION_THERAPY 컬럼 수정
print(pd.unique(x_train["RADIATION_THERAPY"]))
x_train["RADIATION_THERAPY"] = x_train["RADIATION_THERAPY"].fillna("No")

for i in range(0, len(x_train)):
    if x_train["RADIATION_THERAPY"][i] == "Yes":
        x_train["RADIATION_THERAPY"][i] = 0
    else:
        x_train["RADIATION_THERAPY"][i] = 1

# IN_PANCANPATHWAYS_FREEZE 컬럼 수정
print(pd.unique(x_train["IN_PANCANPATHWAYS_FREEZE"]))

for i in range(0, len(x_train)):
    if x_train["IN_PANCANPATHWAYS_FREEZE"][i] == "Yes":
        x_train["IN_PANCANPATHWAYS_FREEZE"][i] = 0
    else :
        x_train["IN_PANCANPATHWAYS_FREEZE"][i] = 1

# 라벨 컬럼
y_train = train_patient[["CANCER_TYPE_ACRONYM"]]
print(pd.unique(y_train["CANCER_TYPE_ACRONYM"]))
cancer_type = ['STAD', 'LUAD', 'LIHC', 'SKCM', 'COAD', 'LGG']

for i in range(0, len(y_train)):
    y_train["CANCER_TYPE_ACRONYM"][i] = cancer_type.index(y_train["CANCER_TYPE_ACRONYM"][i])

# 검증용 데이터
## 학습용 데이터 중 밑에서부터 100개 사용
x_valid = x_train[-100:]
x_train = x_train[:-100]

y_valid = y_train[-100:]
y_train = y_train[:-100]

x_train.isnull().any()
x_valid.isnull().any()

# 테스트 데이터
## 학습용 데이터 생성과 동일한 과정으로 생성
x_test = test_patient[["AGE", "SEX", "WEIGHT", "AJCC_PATHOLOGIC_TUMOR_STAGE", "AJCC_STAGING_EDITION",
    "PATH_M_STAGE", "PATH_N_STAGE", "PATH_T_STAGE", "PERSON_NEOPLASM_CANCER_STATUS",
    "PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT", "RADIATION_THERAPY", "IN_PANCANPATHWAYS_FREEZE"]]

# AGE 컬럼 수정
print(x_test.AGE.describe())
print(pd.unique(x_test.AGE))
x_test.AGE = x_test.AGE.fillna(np.round(np.mean(x_test.AGE)))
x_test["AGE"] = x_test["AGE"].astype("int32")

# SEX 컬럼 수정
x_test["SEX"].describe()
x_test["SEX"] = x_test["SEX"].fillna("Male")

for i in range(0, len(x_test["SEX"])):
        if x_test["SEX"][i] == "Male":
            x_test["SEX"][i] = 0
        elif x_test["SEX"][i] == "Female":
            x_test["SEX"][i] = 1

# WEIGHT
print(x_test.WEIGHT.describe())
print(pd.unique(x_test.WEIGHT))

x_test.WEIGHT = x_test.WEIGHT.fillna(np.mean(x_test.WEIGHT))

# AJCC_PATHOLOGIC_TUMOR_STAGE 컬럼 수정
pd.unique(x_test.AJCC_PATHOLOGIC_TUMOR_STAGE)
x_test.AJCC_PATHOLOGIC_TUMOR_STAGE = x_test.AJCC_PATHOLOGIC_TUMOR_STAGE.fillna("STAGE 0")

stage_level = ['STAGE 0', 'STAGE I', 'STAGE IA', 'STAGE IB', 'STAGE I/II (NOS)',
 'STAGE II', 'STAGE IIA', 'STAGE IIB', 'STAGE IIC',
 'STAGE III', 'STAGE IIIA', 'STAGE IIIB', 'STAGE IIIC',
 'STAGE IV', 'STAGE IVA', 'STAGE IVB']

for i in range(0, len(x_test)):
    if x_test["AJCC_PATHOLOGIC_TUMOR_STAGE"][i] in stage_level:
        x_test["AJCC_PATHOLOGIC_TUMOR_STAGE"][i] = stage_level.index(x_test["AJCC_PATHOLOGIC_TUMOR_STAGE"][i])

# AJCC_STAGING_EDITION 컬럼 수정
pd.unique(x_test["AJCC_STAGING_EDITION"])
x_test["AJCC_STAGING_EDITION"] = x_train["AJCC_STAGING_EDITION"].fillna("unknown")
stage_edition = ['1ST', '2ND', '3RD', '4TH', '5TH', '6TH', '7TH', 'unknown']
for i in range(0, len(x_test)):
    if x_test["AJCC_STAGING_EDITION"][i] in stage_edition:
        x_test["AJCC_STAGING_EDITION"][i] = stage_edition.index(x_test["AJCC_STAGING_EDITION"][i])

# PATH_STAGE 컬럼들 수정
print(pd.unique(x_test["PATH_M_STAGE"]))
print(pd.unique(x_test["PATH_N_STAGE"]))
print(pd.unique(x_test["PATH_T_STAGE"]))

x_test["PATH_M_STAGE"] = x_test["PATH_M_STAGE"].fillna("unknown")
x_test["PATH_N_STAGE"] = x_test["PATH_N_STAGE"].fillna("unknown")
x_test["PATH_T_STAGE"] = x_test["PATH_T_STAGE"].fillna("unknown")

path_m = ["M0", "M1", "M1A", "M1B", "M1C", "MX", "unknown"]
path_n = ["N0", "N1", "N1A", "N1B", "N1C", "N2", "N2A", "N2B", "N2C", "N3", "N3A", "N3B", "NX", "unknown"]
path_t = ["T0", "T1", "T1A", "T1B", "T2", "T2A", "T2B", "T3", "T3A", "T3B", "T4", "T4A", "T4B", "TX","TIS", "unknown"]

for i in range(0, len(x_test)):
    if x_test["PATH_M_STAGE"][i] in path_m:
        x_test["PATH_M_STAGE"][i] = path_m.index(x_test["PATH_M_STAGE"][i])

    if x_test["PATH_N_STAGE"][i] in path_n:
        x_test["PATH_N_STAGE"][i] = path_n.index(x_test["PATH_N_STAGE"][i])

    if x_test["PATH_T_STAGE"][i] in path_t:
        x_test["PATH_T_STAGE"][i] = path_t.index(x_test["PATH_T_STAGE"][i])

# PERSON_NEOPLASM_CANCER_STATUS 컬럼 수정
print(pd.unique(x_test.PERSON_NEOPLASM_CANCER_STATUS))
x_test["PERSON_NEOPLASM_CANCER_STATUS"] = x_test["PERSON_NEOPLASM_CANCER_STATUS"].fillna("Tumor Free")

for i in range(0, len(x_test)):
    if x_test["PERSON_NEOPLASM_CANCER_STATUS"][i].__eq__("Tumor Free"):
        x_test["PERSON_NEOPLASM_CANCER_STATUS"][i] = 0
    else :
        x_test["PERSON_NEOPLASM_CANCER_STATUS"][i] = 1

# PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT 컬럼 수정
pd.unique(x_test["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"])
x_test["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"] = x_test["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"].fillna("No")

for i in range(0, len(x_test)):
    if x_test["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"][i] == "Yes":
        x_test["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"][i] = 0
    else :
        x_test["PRIMARY_LYMPH_NODE_PRESENTATION_ASSESSMENT"][i] = 1

# RADIATION_THERAPY 컬럼 수정
print(pd.unique(x_test["RADIATION_THERAPY"]))
x_test["RADIATION_THERAPY"] = x_test["RADIATION_THERAPY"].fillna("No")

for i in range(0, len(x_test)):
    if x_test["RADIATION_THERAPY"][i] == "Yes":
        x_test["RADIATION_THERAPY"][i] = 0
    else:
        x_test["RADIATION_THERAPY"][i] = 1

# IN_PANCANPATHWAYS_FREEZE 컬럼 수정
print(pd.unique(x_test["IN_PANCANPATHWAYS_FREEZE"]))

for i in range(0, len(x_test)):
    if x_test["IN_PANCANPATHWAYS_FREEZE"][i] == "Yes":
        x_test["IN_PANCANPATHWAYS_FREEZE"][i] = 0
    else :
        x_test["IN_PANCANPATHWAYS_FREEZE"][i] = 1

x_test.isnull().any()

# 모델링
## 데이터 셋 구성
x_train = x_train.values.astype("float32")
y_train = K.utils.to_categorical(y_train.astype("float32"), num_classes=6)

x_valid = x_valid.values.astype("float32")
y_valid = K.utils.to_categorical(y_valid.astype("float32"), num_classes=6)

train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(buffer_size=50).batch(32)
valid_dataset = tf.data.Dataset.from_tensor_slices((x_valid, y_valid)).batch(100)

x_test = x_test.values.astype("float32")


print(x_train.shape)
print(y_train.shape)
print(train_dataset)

## 모델링
model1 = tf.keras.models.Sequential([
    tf.keras.layers.Dense(units=48, activation="relu", input_shape=(12,)),
    tf.keras.layers.Dense(units=24, activation="relu"),
    tf.keras.layers.Dense(units=12, activation="relu"),
    tf.keras.layers.Dense(units=6, activation="softmax")
])
model1.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.1),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)
print(model1.summary())

## 학습하기
history = model1.fit(train_dataset, epochs=100, validation_data=valid_dataset, validation_steps=1)
model1.trainable_variables

id = list(test_patient["id"])
result = np.argmax(model1.predict(x_test), -1)

print(id)
print(result)

resultDF = pd.DataFrame(list(zip(id, result)), columns=["id", "expected"])
print(resultDF)
resultDF.to_csv("data/Cancer_Classification/submission.csv", index=False)