import random
import pdb
import os
from datetime import datetime as dt
import statistics
import math
from argparse import ArgumentParser

import torch
import torch.nn.functional as F
torch.nn.functional
import numpy as np


import torchvision.transforms as transform
from torch.autograd import Variable
from torchvision import transforms
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import BinaryFBetaScore
import matplotlib.pyplot as plt
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import mean_absolute_error

from asset import crf_refine, visialize,convert_tensor_to_pil, img_transform, hsv_transform
from metrics import get_maxFscore_and_threshold


def image_mask_path(data_dir):
    return os.path.join(data_dir, "image"), os.path.join(data_dir, "mask")


def check_strings(string, string_list):
    for item in string_list:
        if item in string:
            return True
    return False


def train(dataloader, model, loss_fn, metrics_fn, optimizer):
    model.train()
    train_loss = 0
    train_score = 0
    train_refine_loss = 0
    for image, mask, edge, _, sv in tqdm(dataloader):
        optimizer.zero_grad()
        input = (image.cuda(), sv.cuda())
        target = mask.cuda()
        target_edge = edge.cuda() 
        pred = model(input)
        loss = loss_fn(pred, target, target_edge)
        loss.backward()
        
        train_loss += loss.detach().item()
        train_score += metrics_fn(torch.sigmoid(pred[-1]).cpu(), mask.int()).item()
        train_refine_loss += loss.detach().item()
        
        optimizer.step()
    # Average loss
    m_loss = train_loss / len(dataloader)                              
    m_score = train_score / len(dataloader)  
    m_refine_loss = train_refine_loss / len(dataloader)
    print(f"train_data\n loss: {m_loss:.5f}, refine_loss:{m_refine_loss:.5f}, score: {m_score:.5f}\n")
    return {"mean_loss": m_loss,
            "mean_metrics": m_score,
            "mean_refine_loss": m_refine_loss
            }


def val(dataloader, model, loss_fn, metrics_fn, output_path):
    val_loss = 0
    val_score = 0
    val_refine_loss = 0
    with torch.no_grad():
        for image, mask, edge, rgb_img, sv in tqdm(dataloader):  
            input = (image.cuda(), sv.cuda())            
            pred = model(input)
            target = mask.cuda()
            target_edge = edge.cuda()
            
            loss = loss_fn(pred, target, target_edge)  
            
            val_loss += loss.detach().item()   
            val_score += metrics_fn(torch.sigmoid(pred[-1]).cpu(), mask.int()).item()
            val_refine_loss += loss.detach().item()

    m_loss = val_loss / len(dataloader)
    m_score = val_score / len(dataloader)
    m_refine_loss = val_refine_loss / len(dataloader)
    print(f"val_data\n loss: {m_loss:.5f}, refine_loss:{m_refine_loss:.5f}, score: {m_score:.5f}\n")
    # Plot graghs
    num_graghs = len(pred)
    col = 4
    row = math.ceil((num_graghs + 3) / col)
    font_size = 36
    plt.figure(tight_layout=True, figsize=(20,30))
    
    for i in range(num_graghs):
        pred_arr = torch.sigmoid(pred[i]).cpu()
        imgobject_pred = convert_tensor_to_pil(pred_arr)
        plt.subplot(row, col, i + 1)
        if i < num_graghs:
            plt.imshow(imgobject_pred, cmap='gray')
        if i < 4:
            plt.title(f"RCCL: {4 - i}", fontsize=font_size)
        elif i < 8:
            plt.title(f"SSF: {8 - i}", fontsize=font_size)
        elif i < 12:
            plt.title(f"SH: {12 - i}", fontsize=font_size)
        elif i == 12:
            plt.title(f"boundary", fontsize=font_size)
        elif i == 13:
            plt.title(f"final", fontsize=font_size)
        else:
            pass
        
    plt.subplot(row, col, num_graghs + 1)
    plt.imshow(rgb_img.squeeze(0).permute(1, 2, 0).contiguous())
    plt.title("image", fontsize=font_size)
    plt.subplot(row, col, num_graghs + 2)
    plt.imshow(mask.view(416, 416).cpu(), cmap='gray')
    plt.title("mask", fontsize=font_size)
    plt.subplot(row, col, num_graghs + 3)
    # Making masked image.
    pred = pred_arr.squeeze(0).permute(1,2,0).numpy()
    rgb_img = rgb_img.squeeze(0).permute(1,2,0).numpy().astype(np.uint8)
    masking_img = (rgb_img * pred).astype(np.uint8)
    plt.imshow(masking_img)
    plt.title("detected mirror", fontsize=font_size)
    plt.axis('off')
    plt.savefig(output_path)
    plt.close()
    return {"mean_loss": m_loss,
            "mean_metrics": m_score,
            "mean_refine_loss": m_refine_loss
            }


def load_and_process_image(img_path, img_trans=img_transform(), hsv_trans=hsv_transform()) -> tuple:
    # Load image
    img = Image.open(img_path)
    # Variable
    w, h = img.size
    img_var = Variable(img_trans(img).unsqueeze(0)).cuda()
    sv_var = Variable(hsv_trans(img.convert("HSV"))[1:,:,:].unsqueeze(0)).cuda()
    
    return img_var, sv_var


def post_resize_and_convert(output_list, resize=None) -> list:
    if resize is None:
        raise ValueError("No setting resize.")
    processed_outmaps = []
    for i, out_map in enumerate(output_list):
        out_map_cpu = out_map.cpu()
        out_map_resized = F.interpolate(out_map_cpu, size=resize, mode='bilinear', align_corners=True)
        out_map_squeezed = torch.squeeze(out_map_resized)
        out_map_final = (out_map_squeezed * 255.).numpy().astype(np.uint8)
        processed_outmaps.append(out_map_final)
        
    return processed_outmaps


def test(test_imgs, mask_dir, model, save_dir, read_best_path):
    model.load_state_dict(torch.load(read_best_path))
    print(f"Load paramters -> {read_best_path}")
    model.eval()
    
    img_trans = img_transform()
    hsv_trans = hsv_transform()

    max_Fbeta_list = []
    MAE_list = []
    with torch.no_grad():
        for img_path in tqdm(test_imgs):
            print(f"image name: {img_path}")
            
            # if os.path.basename(img_path) != "CIMG7899.jpg":
            #     continue
            
            # Load image
            img = Image.open(img_path)
            filename = os.path.basename(img_path).replace(".jpg", ".png")
            mask = Image.open(os.path.join(mask_dir, filename))
            gt = np.array(mask.convert('1')).astype(int)
            # Variable
            w, h = img.size
            img_var = img_trans(img).unsqueeze(0).cuda()
            sv_var = hsv_trans(img.convert("HSV"))[1:,:,:].unsqueeze(0).cuda()
            # Predict
            origin_output_list = list(model((img_var.cuda(), sv_var.cuda())))
            # origin_output_list = [torch.sigmoid(out_map) for out_map in origin_output_list]
            
            # for i in range(len(origin_output_list)):
            #     print(f"origin_output_list[{i}_sig][400, 400]]: {origin_output_list[i][:, :, 400, 400]}")

            # print("Raw value.")
            # for i in range(len(origin_output_list)):
            #     origin_output_list[i] = torch.sigmoid(origin_output_list[i])
            #     print(f"origin_output_list[{i}_raw][400, 400]: {origin_output_list[i][:, :, 400, 400]}")
            
            
            # Post processing
            output_list = post_resize_and_convert(origin_output_list, resize=(h, w))
            print("Predict...")
            
            # pdb.set_trace()

            # Save final predict map
            output_file_path = os.path.join(save_dir, filename)
            print(f"file name: {output_file_path}")
            Image.fromarray(output_list[-1]).save(output_file_path)
            masking = (np.array(img.convert("RGB")) / 255.) * (output_list[-1][:,:,np.newaxis] / 255.)

            # Calucation scores
            pred_1d = (output_list[-1] / 255.).flatten()
            true_1d = gt.flatten()
            max_Fscore, thres = get_maxFscore_and_threshold(true_1d, pred_1d)
            max_Fbeta_list.append(max_Fscore)
            MAE_list.append(mean_absolute_error(true_1d, pred_1d))
            thres_final = np.zeros_like(output_list[-1])
            thres_final[output_list[-1] > (thres * 255.)] = 255
            # Gragh plot
            all_images = output_list + [img, mask, masking, thres_final]
            num_images = len(all_images)
            
            col = 4
            row = math.ceil(num_images / col)
            font_size = 36
            plt.figure(tight_layout=True, figsize=(20, 30))
            
            for i in range(num_images):
                plt.subplot(row, col, i + 1)
                title = ""
                # plot image
                if (i == 14) or (i == 16):
                    plt.imshow(all_images[i])
                else:
                    plt.imshow(all_images[i], cmap="gray")
                # write titles
                if i < 4:
                    title = f"RCCL-{4 - i}"
                elif i < 8:
                    title = f"SSF-{8 - i}"
                elif i < 12:
                    title = f"SH-{12 - i}"
                elif i == 12:
                    title = "boundary"
                elif i == 13:
                    title = "final"
                elif i == 14:
                    title = "image"
                elif i == 15:
                    title = "mask"
                elif i == 16:
                    title = "masked image"
                elif i == 17:
                    title = "best F_score threshold"
                plt.title(title, fontsize=font_size)
                plt.axis('off')
            
            # Save analysis map 
            current_time = dt.now().strftime('%m-%d-%H')
            output_file_name = current_time + "_" + os.path.basename(img_path).replace(".jpg", "_featmap.png")
            output_analysis_path = os.path.join(save_dir, output_file_name)
            print(f"analysis_file: {output_analysis_path}")
            plt.savefig(output_analysis_path)
            plt.close()
            
            # pdb.set_trace()

        avg_f_beta = statistics.mean(max_Fbeta_list)
        avg_MAE = statistics.mean(MAE_list)
        std_fbeta = statistics.pvariance(max_Fbeta_list)
        std_MAE = statistics.pvariance(MAE_list)
        print(f"Max Fbeta: {avg_f_beta:.5f}, MAE: {avg_MAE:.5f}")
        print(f"Std Max Fbeta: {std_fbeta:.5f}, Std MAE: {std_MAE:.5f}")
        return avg_f_beta, avg_MAE


def plot_loss_gragh(t_loss, v_loss, t_score, v_score, save_path):
    fig = plt.figure(figsize=(25,9))
    ax1 = fig.add_subplot(1,2,1)
    ax1.set_title("Loss", fontsize=18)
    ax1.set_xlabel("Epoch",fontsize=18)
    ax1.set_ylabel("Loss",fontsize=18)
    ax1.plot(t_loss, label="train", marker='o')
    ax1.plot(v_loss, label="valid", marker='o')
    ax1.tick_params(axis='both',labelsize=15)
    ax1.grid()
    ax1.legend()
    ax2 = fig.add_subplot(1,2,2)
    ax2.set_title("BinaryFbetaScore", fontsize=18)
    ax2.set_xlabel("Epoch", fontsize=18)
    ax2.set_ylabel("IoU", fontsize=18)
    ax2.plot(t_score, label="train", marker='o')
    ax2.plot(v_score, label="valid", marker='o')
    ax2.tick_params(axis='both',labelsize=15)
    ax2.grid()
    ax2.legend()
    plt.savefig(save_path)
    plt.close()


def model_setting(model, rccl_freeze=False, ssf_freeze=False, sh_freeze=False, EDF_freeze=False):
    print(f"freeze_module:\n RCCL:{rccl_freeze}, SSF:{ssf_freeze}, SH:{sh_freeze} ")
    for name, param in model.named_parameters():
        if ('rccl_' in name):
            if rccl_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif ('ssf_' in name):
            if ssf_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif ('sh_' in name):
            if sh_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        elif ('edge_' in name):
            if EDF_freeze:
                param.requires_grad = False
            else:
                param.requires_grad = True
        else:
            pass
        # print(name, param.requires_grad)
    return model


def model_param_reading(model, read_path, read_module=['rccl', 'ssf', 'sh']):
    if not os.path.exists(read_path):
        print("No read parameter files.")
        return model
    print(f"Read param: {read_path}")
    state_dict = torch.load(read_path)
    selected_state_dict = {key: value for key, value in state_dict.items() if check_strings(string=key, string_list=read_module)}
    model.load_state_dict(selected_state_dict, strict=False)
    return model


def load_rccl_ssf_sh_and_refine_params(model, rccl_param_path, ssf_param_path, sh_param_path):
    
    param_dict = {}
    param_dict['rccl'] = rccl_param_path
    param_dict['ssf'] = ssf_param_path
    param_dict['sh'] = sh_param_path
    
    for compo_name, param_path in param_dict.items():
        
        if not os.path.exists(param_path):
            raise FileNotFoundError(f"Found no file :{param_path}")    
        
        # Read parameters
        print(f"Load {compo_name} parameter -> {param_path}")
        state_dict = torch.load(param_path)
        
        # Filter parameters that are relevant for the current component
        selected_state_dict = {key: value for key, value in state_dict.items() if check_strings(string=key, string_list=[compo_name])}
        
        # # Add the refinement parameter
        # refine_weight= f'{compo_name}_refine.weight'
        # refine_bias = f"{compo_name}_refine.bias"
        # selected_state_dict[refine_weight] = state_dict['tmp_refinement.weight']
        # selected_state_dict[refine_bias] = state_dict['tmp_refinement.bias']

        model.load_state_dict(selected_state_dict, strict=False)

    return model

# Print unfrozen_parameters.
def print_unfrozen_params(model):
    print("Unfrozen parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Parameter Size: {param.size()}")