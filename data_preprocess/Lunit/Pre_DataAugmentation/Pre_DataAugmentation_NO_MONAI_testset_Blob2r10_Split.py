import os
import numpy as np
import shutil
from skimage import io, segmentation, morphology, exposure
import numpy as np
import tifffile as tif
from tqdm import tqdm
import cv2
join = os.path.join
from PIL import Image
import numpy as np
import pandas as pd
import os
import cv2
import matplotlib.pyplot as plt
import albumentations as at

train_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task761_Lunit_Cell_StainNorm_BlobCell_3_concat_Split/imagesTs"
train_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task761_Lunit_Cell_StainNorm_BlobCell_3_concat_Split/labelsTs"
train_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task761_Lunit_Cell_StainNorm_BlobCell_3_concat_Split/Lunit_Tissue_cutting_Upscaling_255/labelsTs"
train_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task761_Lunit_Cell_StainNorm_BlobCell_3_concat_Split/JustCell_BlobCell_labels_255/labelsTs"

aim_train_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/imagesTs"
aim_train_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/labelsTs"
aim_train_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/Lunit_Tissue_cutting_Upscaling_255/labelsTs"
aim_train_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/JustCell_BlobCell_labels_255/labelsTs"
os.makedirs(aim_train_img_path, exist_ok=True)
os.makedirs(aim_train_label_path, exist_ok=True)
os.makedirs(aim_train_tissue_path, exist_ok=True)
os.makedirs(aim_train_cell_path, exist_ok=True)

img_list = sorted(os.listdir(train_img_path))
label_list = sorted(os.listdir(train_label_path))
tisse_list = sorted(os.listdir(train_tissue_path))
cell_list = sorted(os.listdir(train_cell_path))

# val_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/imagesTs"
# val_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTs"
# val_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/Lunit_Tissue_cutting_Upscaling_255/labelsTs"
# val_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task740_Lunit_Cell_StainNorm_r10_Blob_Cell/labelsTs_circle_255"

        # seg = np.array(Image.open(join(train_img_path, case)))


# # for error label
# error_tissue_path = '/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task761_Lunit_Cell_StainNorm_BlobCell_3_concat_Split/error_cell_prediction/labelsTr'



for case in img_list:
    if case[-4:]==".png":
        print(case)
        img = cv2.cvtColor(cv2.imread(join(train_img_path, case)), cv2.COLOR_BGR2RGB)
        label = cv2.cvtColor(cv2.imread(join(train_label_path, case)), cv2.COLOR_BGR2RGB)
        tissue = cv2.cvtColor(cv2.imread(join(train_tissue_path, case)), cv2.COLOR_BGR2RGB)
        cell = cv2.cvtColor(cv2.imread(join(train_cell_path, case)), cv2.COLOR_BGR2RGB)

        # plt.imshow(img)
        # plt.imshow(tissue)
        label_Gray = cv2.cvtColor(label, cv2.COLOR_RGB2GRAY)
        tissue_Gray = cv2.cvtColor(tissue, cv2.COLOR_RGB2GRAY)
        cell_Gray = cv2.cvtColor(cell, cv2.COLOR_RGB2GRAY)

        Image.fromarray(img).save(join(aim_train_img_path, case))
        Image.fromarray(label_Gray).save(join(aim_train_label_path, case))
        Image.fromarray(tissue_Gray).save(join(aim_train_tissue_path, case))
        Image.fromarray(cell_Gray).save(join(aim_train_cell_path, case))




        # # 좌우 반전
        # aug_horizontal = at.HorizontalFlip(p=1.0)

        # hori_img = aug_horizontal(image=img)['image']
        # hori_label = aug_horizontal(image=label)['image']
        # hori_tissue = aug_horizontal(image=tissue)['image']
        # hori_cell = aug_horizontal(image=cell)['image']

        # hori_label = cv2.cvtColor(hori_label, cv2.COLOR_RGB2GRAY)
        # hori_tissue = cv2.cvtColor(hori_tissue, cv2.COLOR_RGB2GRAY)
        # hori_cell = cv2.cvtColor(hori_cell, cv2.COLOR_RGB2GRAY)

        # Image.fromarray(hori_img).save(join(aim_train_img_path, "hori_"+ case))
        # Image.fromarray(hori_label).save(join(aim_train_label_path, "hori_"+ case))
        # Image.fromarray(hori_tissue).save(join(aim_train_tissue_path, "hori_"+ case))
        # Image.fromarray(hori_cell).save(join(aim_train_cell_path, "hori_"+ case))



        # # 상하 반전
        # aug_vertical = at.VerticalFlip(p=1.0)

        # ver_img = aug_vertical(image=img)['image']
        # ver_label = aug_vertical(image=label)['image']
        # ver_tissue = aug_vertical(image=tissue)['image']
        # ver_cell = aug_vertical(image=cell)['image']

        # ver_label = cv2.cvtColor(ver_label, cv2.COLOR_RGB2GRAY)
        # ver_tissue = cv2.cvtColor(ver_tissue, cv2.COLOR_RGB2GRAY)
        # ver_cell = cv2.cvtColor(ver_cell, cv2.COLOR_RGB2GRAY)

        # Image.fromarray(ver_img).save(join(aim_train_img_path, "ver_"+ case))
        # Image.fromarray(ver_label).save(join(aim_train_label_path, "ver_"+ case))
        # Image.fromarray(ver_tissue).save(join(aim_train_tissue_path, "ver_"+ case))
        # Image.fromarray(ver_cell).save(join(aim_train_cell_path, "ver_"+ case))



        # shift scale rotate
        # scale 1.25
        aug_scale_5 = at.ShiftScaleRotate(shift_limit=0, scale_limit=(-0.25, -0.25), rotate_limit=0, p=1)

        scale_5_img = aug_scale_5(image=img)['image']
        scale_5_label = aug_scale_5(image=label)['image']
        scale_5_tissue = aug_scale_5(image=tissue)['image']
        scale_5_cell = aug_scale_5(image=cell)['image']

        scale_5_label = cv2.cvtColor(scale_5_label, cv2.COLOR_RGB2GRAY)
        scale_5_tissue = cv2.cvtColor(scale_5_tissue, cv2.COLOR_RGB2GRAY)
        scale_5_cell = cv2.cvtColor(scale_5_cell, cv2.COLOR_RGB2GRAY)

        Image.fromarray(scale_5_img).save(join(aim_train_img_path, "scale_5_"+ case))
        Image.fromarray(scale_5_label).save(join(aim_train_label_path, "scale_5_"+ case))
        Image.fromarray(scale_5_tissue).save(join(aim_train_tissue_path, "scale_5_"+ case))
        Image.fromarray(scale_5_cell).save(join(aim_train_cell_path, "scale_5_"+ case))



        # scale 0.25
        aug_scale_25 = at.ShiftScaleRotate(shift_limit=0, scale_limit=(0.25,0.25), rotate_limit=0, p=1)

        scale_25_img = aug_scale_25(image=img)['image']
        scale_25_label = aug_scale_25(image=label)['image']
        scale_25_tissue = aug_scale_25(image=tissue)['image']
        scale_25_cell = aug_scale_25(image=cell)['image']

        scale_25_label = cv2.cvtColor(scale_25_label, cv2.COLOR_RGB2GRAY)
        scale_25_tissue = cv2.cvtColor(scale_25_tissue, cv2.COLOR_RGB2GRAY)
        scale_25_cell = cv2.cvtColor(scale_25_cell, cv2.COLOR_RGB2GRAY)

        Image.fromarray(scale_25_img).save(join(aim_train_img_path, "scale_25_"+ case))
        Image.fromarray(scale_25_label).save(join(aim_train_label_path, "scale_25_"+ case))
        Image.fromarray(scale_25_tissue).save(join(aim_train_tissue_path, "scale_25_"+ case))
        Image.fromarray(scale_25_cell).save(join(aim_train_cell_path, "scale_25_"+ case))




        # #가우시안 노이즈 분포를 가지는 노이즈를 추가
        # aug_noise = at.GaussNoise(p=1, var_limit=(400, 400))

        # GauN_img = aug_noise(image=img)['image']
        # GauN_label = aug_noise(image=label)['image']
        # GauN_tissue = aug_noise(image=tissue)['image']
        # GauN_cell = aug_noise(image=cell)['image']

        # Image.fromarray(GauN_img).save(join(aim_train_img_path, "GauN_"+ case))
        # Image.fromarray(label_Gray).save(join(aim_train_label_path, "GauN_"+ case))
        # Image.fromarray(tissue_Gray).save(join(aim_train_tissue_path, "GauN_"+ case))
        # Image.fromarray(cell_Gray).save(join(aim_train_cell_path, "GauN_"+ case))





        # # blur_limit가 클수록 더 흐림
        # aug_blur = at.Blur(p=1, blur_limit=(5, 5))

        # blur_img = aug_blur(image=img)['image']
        # blur_label = aug_blur(image=label)['image']
        # blur_tissue = aug_blur(image=tissue)['image']
        # blur_cell = aug_blur(image=cell)['image']

        # Image.fromarray(blur_img).save(join(aim_train_img_path, "blur_"+ case))
        # Image.fromarray(label_Gray).save(join(aim_train_label_path, "blur_"+ case))
        # Image.fromarray(tissue_Gray).save(join(aim_train_tissue_path, "blur_"+ case))
        # Image.fromarray(cell_Gray).save(join(aim_train_cell_path, "blur_"+ case))






        # # Cell 모델과 Tissue 모델의 Prediction을 input으로 넣어주기

        # error_cell_label = cv2.imread(join(error_tissue_path, case), -1)

        # Image.fromarray(img).save(join(aim_train_img_path, "error_"+ case))
        # Image.fromarray(label_Gray).save(join(aim_train_label_path, "error_"+ case))
        # Image.fromarray(tissue_Gray).save(join(aim_train_tissue_path, "error_"+ case))
        # Image.fromarray(error_cell_label).save(join(aim_train_cell_path, "error_"+ case))
        








