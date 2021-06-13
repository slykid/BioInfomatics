import pandas as pd
import numpy as np

#  train dataset
train_patient = pd.read_csv("data/Cancer_Classification/train_data_clinical_patient.csv")
train_cna = pd.read_csv("data/Cancer_Classification/train_data_cna.csv")
train_mutation = pd.read_csv("data/Cancer_Classification/train_data_mutations.csv")
train_rna = pd.read_csv("data/Cancer_Classification/train_data_rna-seq.csv")

# test dataset
test_patient = pd.read_csv("data/Cancer_Classification/test_data_clinical_patient_v1.csv")
test_cna = pd.read_csv("data/Cancer_Classification/test_data_cna_v1.csv")
test_mutation = pd.read_csv("data/Cancer_Classification/test_data_mutations_v1.csv")
test_rna = pd.read_csv("data/Cancer_Classification/test_data_rna-seq_v1.csv")

# 학습데이터
train_patient.isnull().any()
train_patient.info()
train_patient.describe()
