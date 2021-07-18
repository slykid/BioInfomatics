import os
import pandas as pd
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

# 이미지 처리
## 1) 이미지 처리 : dcm -> jpg
train_dir = "data\\SIIM_FISABIO_RSNA\\train"
test_dir = "data\\SIIM_FISABIO_RSNA\\test"

def read_xray(path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.read_file(path)

    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if fix_monochrome and dicom.PhotometricInterpretation == "MONOCHROME1":
        data = np.amax(data) - data

    data = data - np.min(data)
    data = data / np.max(data)
    data = (data * 255).astype(np.uint8)

    return data

def resize(array, size, keep_ratio=False, resample=Image.LANCZOS):
    im = Image.fromarray(array)

    if keep_ratio:
        im.thumbnail((size, size), resample)
    else:
        im = im.resize((size, size), resample)

    return im

image_id = []
dimension0 = []
dimension1 = []
splits = []

for split in ['train', 'test']:
    save_dir = f'data/SIIM_FISABIO_RSNA/prep/{split}/'
    os.makedirs(save_dir, exist_ok=True)

    for dirname, _, filenames in tqdm(os.walk(f'data/SIIM_FISABIO_RSNA/{split}')):
        for file in filenames:
            xray = read_xray(os.path.join(dirname, file))
            im = resize(xray, size=512)  ## resize to 512
            im.save(os.path.join(save_dir, file.replace('dcm', 'jpg')))

            image_id.append(file.replace('.dcm', ''))
            dimension0.append(xray.shape[0])
            dimension1.append(xray.shape[1])
            splits.append(split)

df = pd.DataFrame.from_dict({'image_id': image_id, 'dim0': dimension0, 'dim1': dimension1, 'split': splits})
df.to_csv('data/SIIM_FISABIO_RSNA/prep/meta.csv', index=False)