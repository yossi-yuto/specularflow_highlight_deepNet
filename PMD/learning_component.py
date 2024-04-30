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
import matplotlib.pyplot as plt


import torchvision.transforms as transform

# from asset import crf_refine, visialize,convert_tensor_to_pil, img_transform, hsv_transform
# from metrics import get_maxFscore_and_threshold


def image_mask_path(data_dir):
    return os.path.join(data_dir, "image"), os.path.join(data_dir, "mask")


def check_strings(string, string_list):
    for item in string_list:
        if item in string:
            return True
    return False



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


def load_weights_partial(model, param_path, component='rccl'):
    saved_state_dict = torch.load(param_path)
    selected_state_dict = {k: v for k, v in saved_state_dict.items() if component in k}
    model.load_state_dict(selected_state_dict, strict=False)
    return model

# Print unfrozen_parameters.
def print_unfrozen_params(model):
    print("Unfrozen parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"Layer: {name}, Parameter Size: {param.size()}")