import os, tempfile
import random
import torch
import wandb
import numpy as np
import argparse as ap
import torch.nn as nn
import monai
from core.call_data import *
from core.call_model import *
from core.core_for_lunit_3_concat_No_MONAI import train
# from core.core_for_tanh import train

from core.train_search import main_search
from config.call import call_config
from transforms.call import call_trans_function
# from core.core_for_crossval import train_for_crossval
# from core.datasets_for_lunit import call_dataloader_Lunit
from core.datasets_for_No_MONAI import call_dataloader_Lunit
join = os.path.join

import warnings
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import random
# torch.multiprocessing.set_start_method('spawn')

# warnings.filterwarnings("ignore")
# torch.multiprocessing.set_start_method('spawn')

def main(info, config, logging=False):
    if logging:
        run = wandb.init(project=info["PROJ_NAME"], entity=info["ENTITY"]) 
        wandb.config.update(config) 

    torch.manual_seed(config["SEEDS"])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config["SEEDS"])
    # random.seed(config["SEEDS"])

    
    # Dataset
    # train_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/imagesTr"
    # train_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/labelsTr"
    # train_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/Lunit_Tissue_cutting_Upscaling_200_100/labelsTr"
    # train_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/labelsTr_circle_255"

    # val_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/imagesTs"
    # val_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/labelsTs"
    # val_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/Lunit_Tissue_cutting_Upscaling_200_100/labelsTs"
    # val_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task745_Lunit_Cell_StainNorm_r10_Blob_Cell_Aug/labelsTs_circle_255"


    train_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task763_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/imagesTr"
    train_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task763_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/labelsTr"
    train_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task763_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/Lunit_Tissue_cutting_Upscaling_255/labelsTr"
    train_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task763_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/JustCell_BlobCell_labels_255/labelsTr"

    val_img_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/imagesTs"
    val_label_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/labelsTs"
    val_tissue_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/Lunit_Tissue_cutting_Upscaling_255/labelsTs"
    val_cell_path = "/vast/AI_team/sukmin/datasets/Lunit_Challenge/Task762_Lunit_Cell_StainNorm_BlobCell_3_concat_Split_Aug/JustCell_BlobCell_labels_255/labelsTs"


    model = call_model(info, config)
    # model = nn.DataParallel(model)
    if logging:
        wandb.watch(model, log="all")
    model.to("cuda")
    # from torchsummary import summary
    # summary(model, (3, 512, 512))
    # print(model)

    optimizer = call_optimizer(config, model)

    if config["LOAD_MODEL"]:
        check_point = torch.load(os.path.join(info["LOGDIR"], f"model_best.pth"))
        try:
            model.load_state_dict(check_point['model_state_dict'])
            optimizer.load_state_dict(check_point['optimizer_state_dict'])
        except:
            model.load_state_dict(check_point)

    best_loss = 1.
    global_step = 0
    dice_val_best = 0.0
    steps_val_best = 0.0
    
    dice_val_best_class = 0.0
    steps_val_best_class = 0.0



    if info["FOLD_for_CrossValidation"] == False:
        train_loader = call_dataloader_Lunit(info, config, train_img_path, train_label_path, train_tissue_path, train_cell_path, transforms=None, mode="train")
        valid_loader = call_dataloader_Lunit(info, config, val_img_path, val_label_path, val_tissue_path, val_cell_path, transforms=None, mode="valid")
        
        
        # valid_loader = call_dataloader(info, config, valid_list, val_transforms, progress=True, mode="valid")
        # check_data = monai.utils.misc.first(valid_loader)
        # print(
        #     "sanity check:",
        #     check_data["image"].shape,
        #     torch.max(check_data["image"]),
        #     check_data["label"].shape,
        #     torch.max(check_data["label"]),
        # )
        torch.manual_seed(config["SEEDS"])
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        np.random.seed(config["SEEDS"])
        
        while global_step < config["MAX_ITERATIONS"]:
            global_step, dice_val_best, steps_val_best, dice_val_best_class, steps_val_best_class, train_loss = train(
                info, config, global_step, dice_val_best, steps_val_best,
                dice_val_best_class, steps_val_best_class, model, optimizer,
                train_loader, valid_loader, 
                logging, deep_supervision=info["Deep_Supervision"],
            )   
            if logging:
                wandb.log({
                    'train_loss': train_loss,
                    # 'train_dice': train_dice,
                }) 
    

    # else:
    #     train_loader = call_dataloader_Crossvalidation(info, config, train_list, train_transforms, progress=True, mode="train")
    #     valid_loader = call_dataloader_Crossvalidation(info, config, valid_list, val_transforms, progress=True, mode="valid")

    #     check_data = monai.utils.misc.first(train_loader[0])
    #     print(
    #         "sanity check:",
    #         check_data["image"].shape,
    #         torch.max(check_data["image"]),
    #         check_data["label"].shape,
    #         torch.max(check_data["label"]),
    #     )

    #     while global_step < config["MAX_ITERATIONS"]:
    #         global_step, dice_val_best, steps_val_best, train_loss = train_for_crossval(
    #             info, config, global_step, dice_val_best, steps_val_best,
    #             model, optimizer,
    #             train_loader, valid_loader, 
    #             logging, deep_supervision=info["Deep_Supervision"],
    #         )   
    #         if logging:
    #             wandb.log({
    #                 'train_loss': train_loss,
    #                 # 'train_dice': train_dice,
    #             })



    if logging:
        artifact = wandb.Artifact('model', type='model')
        # artifact.add_file(
        #     os.path.join(info["LOGDIR"], f"model_best.pth"), 
        #     name=f'model/{config["MODEL_NAME"]}')
        run.log_artifact(artifact)
    return 

    

if __name__ == "__main__":

    parser = ap.ArgumentParser()
    parser.add_argument('-trainer', default= 'Lunit_3_concat_Tissue_Cell_r10_with_StainNorm_Blob_Cell_No_MONAI_BlobCell_to_r10_with_error') # Stroma_200x_only_Epi
    args = parser.parse_args()
    info, config = call_config(args.trainer)
    warnings.filterwarnings("ignore")

    os.environ["CUDA_VISIBLE_DEVICES"] = info["GPUS"]
    main(info, config, logging=True)

