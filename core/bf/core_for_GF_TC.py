import os, sys, glob
# sys.path.append('../')
import torch
import wandb
import numpy as np
from torch import nn
from tqdm import tqdm

from monai.data import *
from monai.metrics import *
from monai.transforms import Activations
from monai.inferers import sliding_window_inference

from core.utils import *   
from core.call_loss import *
from core.call_model import *
from monai.transforms import (
    Activations,
    AsDiscrete,
    Compose,
    EnsureType,
)
import monai
from monai.networks import one_hot

from core.gradient_flow_utils import *
from core.count_TC import *


# post_pred = Compose(
#         [EnsureType(), Activations(softmax=True), AsDiscrete(threshold=0.5)]
#     )
# post_gt = Compose([EnsureType(), AsDiscrete(to_onehot=None)])


def validation(info, config, valid_loader, model, logging=False, threshold=0.5):  
    activation = Activations(sigmoid=True) # softmax : odd result! ToDO : check!  
    dice_metric = DiceMetric(include_background=False, reduction='none')
    confusion_matrix = ConfusionMatrixMetric(include_background=False, reduction='none')

    epoch_iterator_val = tqdm(
        valid_loader, desc="Validate (X / X Steps)", dynamic_ncols=True
    )
    dice_class, mr_class, fo_class = [], [], []
    model.eval()
    with torch.no_grad():
        for step, batch in enumerate(epoch_iterator_val):
            step += 1
            
            val_inputs, val_labels = batch["image"].to("cuda"), batch["label"].to("cuda") # , non_blocking=True).long()

            val_labels = one_hot(val_labels, config["CHANNEL_OUT"])
            val_outputs = sliding_window_inference(val_inputs, config["INPUT_SHAPE"], 4 , model) # , device='cuda', sw_device='cuda')
            val_outputs = val_outputs[0]
            
            epoch_iterator_val.set_description(
                "Validate (%d / %d Steps)" % (step, len(epoch_iterator_val))
            )

            if info["Deep_Supervision"] == True: 
                dice_class.append(dice_metric(val_outputs[0]>=threshold, val_labels)[0])

                confusion = confusion_matrix(val_outputs[0]>=threshold, val_labels)[0]
                mr_class.append([
                    calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])
                fo_class.append([
                    calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])


            else:
                dice_class.append(dice_metric(val_outputs>=threshold, val_labels)[0])

                confusion = confusion_matrix(val_outputs>=threshold, val_labels)[0]
                mr_class.append([
                    calc_confusion_metric('fnr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])
                fo_class.append([
                    calc_confusion_metric('fpr',confusion[i]) for i in range(info["CHANNEL_OUT"]-1)
                ])
            # torch.cuda.empty_cache()
        dice_dict, dice_val = calc_mean_class(info, dice_class, 'valid_dice')
        miss_dict, miss_val = calc_mean_class(info, mr_class, 'valid_miss rate')
        # false_dict, false_val = calc_mean_class(info, fo_class, 'valid_false alarm')
        if info["Deep_Supervision"]==True:
            # print(dice_val)
            # print(dice_val.item())
            wandb.log({'valid_dice': dice_val.item(), 'valid_miss rate': miss_val.item(),
            # 'valid_image': log_image_table(info, val_inputs[0].cpu(),
                                            # val_labels[0].cpu(), val_outputs[0][0].cpu())
                                            })
            wandb.log(dice_dict)
            
        else:
            # print(dice_val)
            # print(dice_val.item())
            wandb.log({'valid_dice': dice_val.item(), 'valid_miss rate': miss_val.item(),
            # 'valid_image': log_image_table(info, val_inputs[0].cpu(),
            #                                 val_labels[0].cpu(), val_outputs[0].cpu())
            })                            
        # })

            wandb.log(dice_dict)
            # wandb.log(miss_dict)
            # wandb.log(false_dict)        
    return dice_val



def train(info, config, global_step, dice_val_best, steps_val_best, model, optimizer, train_loader, valid_loader, logging=False, deep_supervision=True): 
    # print(model)
    loss_function = call_loss(loss_mode=config["LOSS_NAME"], sigmoid=True, config=config)
    dice_loss_f = call_loss(loss_mode='dice', sigmoid=True)
    dicece_loss = call_loss(loss_mode='dicece', sigmoid=True)
    ce_loss_f = call_loss(loss_mode='ce', sigmoid=True)
    mse_loss_f = nn.MSELoss(reduction="mean")
    focal_loss_f = call_loss(loss_mode='focal', sigmoid=True)
 
    model.train()

    step = 0
    epoch_loss, epoch_dice = 0., 0.
    print(train_loader)
    epoch_iterator = tqdm(
        train_loader, desc="Training (X / X Steps) (loss=X.X)", dynamic_ncols=True
    )
    print(epoch_iterator)
    for step, batch in enumerate(epoch_iterator):
        step += 1

        x, y1 = batch["image"].to("cuda"), batch["label"].to("cuda")
        # logit_map = model(x)
        logit_map, GF = model(x)

        y = one_hot(
                y1, config["CHANNEL_OUT"]
            )  # (b,cls,256,256)

        # Map label masks to graidnet and onehot
        device ="cuda:0"
        f = labels_to_flows(
            y1, use_gpu=True, device=device
        )
        f = torch.from_numpy(f).float().to(device)




        # count TC
        Yt = TC_value(y1)
        logit_map_soft = torch.nn.functional.softmax(logit_map, dim=1) # (B, C, H, W)
        Xt = TC_value_for_logit_map(logit_map_soft)
        TC_loss = torch.tensor(calculate_TC_loss(Yt, Xt))
        

        loss, dice = 0, 0
        if deep_supervision == True: # deep supervision
            for ds in logit_map:
                # loss += loss_function(ds, y)
                ce_loss = ce_loss_f(ds, y)
                dice_loss = dice_loss_f(ds, y)
                loss += ce_loss + dice_loss
                dice += 1 - dice_loss

            loss /= len(logit_map)
            dice /= len(logit_map)
        else:
            loss_f = loss_function(logit_map, y)
            GF_loss = mse_loss_f(GF, f)
            # TC_loss = mse_loss_f(Yt, Xt)

            loss = (loss_f) + (GF_loss * 0.3) + (TC_loss * 0.3)

        epoch_loss += loss.item()
        # epoch_dice += dice.item()
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        epoch_iterator.set_description(
            "Training (%d / %d Steps) (loss=%2.5f)" % (global_step+1, config["MAX_ITERATIONS"], loss)
        )

        # 0.25k마다 model save
        if global_step % 250 == 0 and global_step != 0 :
            torch.save({
                "global_step": global_step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
            }, os.path.join(info["LOGDIR"], "model_e{0:05d}.pth".format(global_step)))
            print(
                f"Model Was Saved ! steps: {global_step}"
            )


        if (
            global_step % config["EVAL_NUM"] == 0 and global_step != 0
        ) or global_step == config["MAX_ITERATIONS"]:
            
            dice_val = validation(info, config, valid_loader, model, logging)

            if dice_val > dice_val_best:
                dice_val_best = dice_val
                steps_val_best = global_step
                torch.save({
                    "global_step": global_step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                }, os.path.join(info["LOGDIR"], f"model_best_{global_step}.pth"))
                print(
                    f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
                )
            else:
                print(
                    f"Model Was Not Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val} Best model step is {steps_val_best}" 
                )

            # # 1k마다 model save
            # if global_step % 525 == 0 and global_step != 0 :
            #     torch.save({
            #         "global_step": global_step,
            #         "model_state_dict": model.state_dict(),
            #         "optimizer_state_dict": optimizer.state_dict(),
            #     }, os.path.join(info["LOGDIR"], "model_e{0:05d}.pth".format(global_step)))
            #     print(
            #         f"Model Was Saved ! Current Best Avg. Dice: {dice_val_best} Current Avg. Dice: {dice_val}"
            #     )

        global_step += 1
    return global_step, dice_val_best, steps_val_best, epoch_loss / step # , epoch_dice / step
