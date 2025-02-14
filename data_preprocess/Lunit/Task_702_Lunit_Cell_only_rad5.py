# re-read_img.py를 실행 해줘야만 라벨을 RGB로 인지한다
# https://github.com/open-mmlab/mmsegmentation/issues/550

import shutil
import os
import json
import tifffile as tif
from skimage import io, segmentation, morphology, exposure
# from batchgenerators.utilities.file_and_folder_operations import *
join = os.path.join
import numpy as np

def save_json(obj, file: str, indent: int = 4, sort_keys: bool = True) -> None:
    with open(file, 'w') as f:
        json.dump(obj, f, sort_keys=sort_keys, indent=indent)

if __name__ == "__main__":
    """
    This is the code from sukmin Ha
    """

    base = "/vast/AI_team/sukmin/datasets/Lunit_Norm"

    task_id = 702
    task_name = "Lunit_Cell_only_rad5"
    foldername = "Task%03.0d_%s" % (task_id, task_name)

    nnUNet_raw_data = '/vast/AI_team/sukmin/datasets'

    out_base = join(nnUNet_raw_data, foldername)

    imagestr = join(out_base, "imagesTr")
    imagests = join(out_base, "imagesTs")
    labelstr = join(out_base, "labelsTr")
    labelsts = join(out_base, "labelsTs")

    os.makedirs(imagestr, exist_ok=True)
    os.makedirs(imagests, exist_ok=True)
    os.makedirs(labelstr, exist_ok=True)
    os.makedirs(labelsts, exist_ok=True)

    train_patient_names = []
    val_patient_names = []
    test_patient_names = []
    img_list = next(os.walk(join(base, 'images', 'train', 'cell_start_0')))[2]
    seg_list = next(os.walk(join(base, 'annotations', 'train', 'cell_seg_5')))[2]


    # train_patients = all_cases[:150] + all_cases[300:540] + all_cases[600:700]
    # test_patients = all_cases[150:209] + all_cases[700:]

    # train, test를 8:2 비율로 나눠줌 (5 fold Cross validation 적용 예정)
    val_len = int(len(img_list) * 0.2)
    train_patients = img_list[val_len:]
    val_patients = img_list[:val_len]


    for p in train_patients:
        if p != "Thumbs.db":
            img_file = join(base, 'images', 'train', 'cell_start_0', p)
            seg_file = join(base, 'annotations', 'train', 'cell_seg_5', p)

            shutil.copy(img_file, join(imagestr, p))
            shutil.copy(seg_file, join(labelstr, p))
            train_patient_names.append(p)

    
    for p in val_patients:
        if p != "Thumbs.db":
            img_file = join(base, 'images', 'train', 'cell_start_0', p)
            seg_file = join(base, 'annotations', 'train', 'cell_seg_5', p)

            shutil.copy(img_file, join(imagests, p))
            shutil.copy(seg_file, join(labelsts, p))
            val_patient_names.append(p)


    # for p in test_patients:
    #     img_file = join(base, 'images', p)
    #     seg_file = join(base, 'labels', p)

    #     shutil.copy(img_file, join(imagests, p))
    #     shutil.copy(seg_file, join(labelsts, p))
    #     test_patient_names.append(p)    




    json_dict = {}
    json_dict['name'] = "Lunit for Challenge"
    json_dict['description'] = "Suk Min Ha"
    json_dict['tensorImageSize'] = "3D"
    json_dict['reference'] = "Seegene MF"
    json_dict['licence'] = ""
    json_dict['release'] = "0.0"
    json_dict['labels'] = {
        "0": "background",
        "1": "BC (Background Cell)",
        "2": "TC (Tumor Cell)"
    }

    json_dict['numTraining'] = len(train_patient_names)
    json_dict['numVal'] = len(val_patient_names)
    json_dict['training'] = [{'image': "./imagesTr/%s" % i.split("/")[-1], "label": "./labelsTr/%s" % i.split("/")[-1]} for i in
                         train_patient_names]
    json_dict['valid'] = [{'image': "./imagesTs/%s" % i.split("/")[-1], "label": "./labelsTs/%s" % i.split("/")[-1]} for i in
                         val_patient_names]

    save_json(json_dict, os.path.join(out_base, "dataset.json"))



# re-read_img.py를 실행 해줘야만 라벨을 RGB로 인지한다
# https://github.com/open-mmlab/mmsegmentation/issues/550
