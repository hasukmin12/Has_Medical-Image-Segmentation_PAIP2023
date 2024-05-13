#%%
from PIL import Image
from platform import python_version
import cv2
import numpy as np
import tifffile as tif


# # test a single image
img = '/vast/AI_team/sukmin/Results_for_PAIP_final/Test_CAD_DL_ConvNext_large_b8_bbbaseline_DiceCE_GF_0.3_r1024_223/te_c010.png'
visual_rst = '/vast/AI_team/sukmin/Results_visualize_PAIP_2023/Val_CAD_DL_ConvNext_large_b16_baseline_b/seg_val_p008.png'
tumor_gt = '/vast/AI_team/sukmin/Results_visualize_seperate_PAIP_2023/Val_CAD_DL_ConvNext_large_b16_baseline_b/c1_val_p008.png'
non_tumor_gt = '/vast/AI_team/sukmin/Results_visualize_seperate_PAIP_2023/Val_CAD_DL_ConvNext_large_b16_baseline_b/c2_val_p008.png'


# img = '/vast/AI_team/sukmin/datasets/1. PAIP2023_Training/tr_p001.png'
# tumor_gt = '/vast/AI_team/sukmin/datasets/2. PAIP2023_Traning masks/tumor/tr_c001_tumor.png'
# non_tumor_gt = '/vast/AI_team/sukmin/datasets/2. PAIP2023_Traning masks/non_tumor/tr_c001_nontumor.png'


# Let's take a look at the dataset
import cv2
import matplotlib.pyplot as plt

img = cv2.imread(img, -1)
plt.figure(figsize=(8, 6))
plt.imshow(cv2.bgr2rgb(img*100))
plt.show()

print(img.shape)
print(img.max())

# gt = cv2.imread(visual_rst, -1)
# plt.figure(figsize=(8, 6))
# plt.imshow(gt)
# plt.show()




# gt = cv2.imread(tumor_gt, -1) 
# plt.figure(figsize=(8, 6))
# plt.imshow(gt)
# plt.show()

# print(gt.shape)
# print(gt.max())
# print(gt.ndim)
# print(gt.dtype)



# rst = cv2.imread(non_tumor_gt, -1)
# plt.figure(figsize=(8, 6))
# plt.imshow(rst)
# plt.show()

# print(rst.shape)
# print(rst.max())
# print(rst.ndim)
# print(rst.dtype)




# %%
