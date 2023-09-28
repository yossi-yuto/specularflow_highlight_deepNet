from tkinter import font
import os
import datetime
import math
import pdb
from datetime import datetime as dt

import numpy as np
import torch
import matplotlib.pyplot as plt
import pydensecrf.densecrf as dcrf

import torchvision.transforms as transform


def img_transform(resize=(416,416)) -> transform.Compose:
    transform_ = transform.Compose([
        transform.ToTensor(),
        transform.Resize(resize),
        transform.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform_


def mask_transform(resize=(416,416))-> transform.Compose:
    transform_ = transform.Compose([
        transform.ToTensor(),
        transform.Resize((resize)),
    ])
    return transform_


def hsv_transform(resize=(416,416))-> transform.Compose:
    transform_ = transform.Compose([
        transform.ToTensor(),
        transform.Resize((resize)),
    ])
    return transform_


def convert_tensor_to_pil(img_tensor):
    # Batch Number is only 1.
    to_pil = transform.ToPILImage()
    
    assert img_tensor.size()[0] == 1, f"batch num is not 1."
    return to_pil(img_tensor.squeeze(0))


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def crf_refine(img, annos):
    assert img.dtype == np.uint8
    assert annos.dtype == np.uint8
    assert img.shape[:2] == annos.shape

    # img and annos should be np array with data type uint8

    EPSILON = 1e-8

    M = 2  # salient or not
    tau = 1.05
    # Setup the CRF model
    d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], M)

    anno_norm = annos / 255.

    n_energy = -np.log((1.0 - anno_norm + EPSILON)) / (tau * _sigmoid(1 - anno_norm))
    p_energy = -np.log(anno_norm + EPSILON) / (tau * _sigmoid(anno_norm))

    U = np.zeros((M, img.shape[0] * img.shape[1]), dtype='float32')
    U[0, :] = n_energy.flatten()
    U[1, :] = p_energy.flatten()

    d.setUnaryEnergy(U)

    d.addPairwiseGaussian(sxy=3, compat=3)
    d.addPairwiseBilateral(sxy=60, srgb=5, rgbim=img, compat=5)

    # Do the inference
    infer = np.array(d.inference(1)).astype('float32')
    res = infer[1, :]

    res = res * 255
    res = res.reshape(img.shape[:2])
    return res.astype('uint8')




def visialize(original_image, mask_image, pred, output_dir, output_file_name, train=True, detected_image=None):
    num_outmaps = len(pred)
    pred.extend([original_image, mask_image])
    if not train:
        pred.append(detected_image)
    num_graghs = len(pred)
    col = 4
    row = math.ceil(num_graghs / col)
    font_size = 36
    plt.figure(tight_layout=True, figsize=(20,30))
    for i in range(num_graghs):
        if train:
            pred[i] = convert_tensor_to_pil(pred[i])
        plt.subplot(row, col, i + 1)
        if i < num_outmaps:
            plt.imshow(pred[i], cmap='gray')
            if i < 4:
                plt.title(f"Mirror Map: {4 - i}", fontsize=font_size)
            elif i < 8:
                plt.title(f"SSF Map: {8 - i}", fontsize=font_size)
            elif i < 12:
                plt.title(f"SH Map: {12 - i}", fontsize=font_size)
            elif i == 12:
                plt.title(f"Boundary Map", fontsize=font_size)
            elif i == 13:
                plt.title(f"Output", fontsize=font_size)
        else:
            if i == 14:
                plt.imshow(pred[i])
                plt.title("RGB Image", fontsize=font_size)
            elif i == 15:
                plt.imshow(pred[i], cmap='gray')
                plt.title("GT", fontsize=font_size)
            elif i == 16:
                plt.imshow(pred[i])
                plt.title("Detected Mirror", fontsize=font_size)
        plt.axis('off')
    output_file_name = dt.now().strftime('%m-%d_%H') + "_"  + output_file_name + "_featmap.png"
    output_path = os.path.join(output_dir, output_file_name)
    plt.savefig(output_path)
    plt.close()
    

def plot_learning_map(img_list, output_dir, save_figname, col=4):

    '''gragh size'''
    num_list = len(img_list)
    row = math.ceil(num_list / col)
    print(f"({row},{col})のグラフを作成します。")
    
    fig, ax = plt.subplots(row, col, figsize=(16, 25), tight_layout=True)
    
    font_size = 15
    count = 0
    for i in range(row):
        for j in range(col):
        
            img = convert_tensor_to_pil(img_list[count])

            #assert img_list[count].ndim() == 4, f"Dimention 4 count:{count}"
            if count < 4:
                ax[i, j].imshow(img, cmap='gray')
                ax[i, j].set_title(f"Mirror Map: {4 - count}", fontsize=font_size)
            elif count < 8:
                ax[i, j].imshow(img, cmap='gray')
                ax[i, j].set_title(f"SSF Map: {8 - count}", fontsize=font_size)
            elif count < 12:
                ax[i, j].imshow(img, cmap='gray')
                ax[i, j].set_title(f"SH Map: {12 - count}", fontsize=font_size)
            elif count == 12:
                ax[i, j].imshow(img, cmap='gray')
                ax[i, j].set_title(f"Boundary Map", fontsize=font_size)
            elif count == 13:
                ax[i, j].imshow(img, cmap='gray')
                ax[i, j].set_title(f"Final Predict Map", fontsize=font_size)
            elif count == 14:
                ax[i, j].imshow(img, cmap='gray')
                ax[i, j].set_title(f"RGB Image", fontsize=font_size)
            elif count == 15:
                ax[i, j].imshow(img)
                ax[i, j].set_title(f"GT", fontsize=font_size)
            elif count == 16:
                ax[i, j].imshow(img)
                ax[i, j].set_title(f"Detected Image", fontsize=font_size)
            elif count == 17:
                ax[i, j].imshow(img)
                ax[i, j].set_title(f"Resize Image", fontsize=font_size)
            ax[i, j].axis('off')
            
            count += 1
        
            if count > num_list - 1:
                break 
    
    img_name = dt.now().strftime('%m-%d_%H') + "_"  + save_figname + "_featmap.png"
    output_file = os.path.join(output_dir, img_name)   
    fig.savefig(output_file, bbox_inched='tight', pad_inches=0, dpi=300)
    print(f"{output_file}に出力")
    plt.close()
    
    