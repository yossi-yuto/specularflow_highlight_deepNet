import os.path as osp
import math
import pdb

import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image


def train(dataloader, model, loss_fn, metrics_fn, optimizer):
    model.train()
    train_loss = []
    train_score = []
    for image, sv, mask, edge, _ in tqdm(dataloader):
        optimizer.zero_grad()
        pred = model((image.cuda(), sv.cuda()))
        loss = loss_fn(pred, mask.cuda(), edge.cuda())
        loss.backward()
        train_loss.append(loss.detach().item())
        train_score.append(metrics_fn(torch.sigmoid(pred[-1]).cpu(), mask.int()).item())
        optimizer.step()
    # Average loss
    m_loss = np.mean(train_loss)                         
    m_score = np.mean(train_score)
    print(f"train_data\n loss: {m_loss:.5f}, score: {m_score:.5f}\n")
    return {"iter_avg_loss": m_loss, "iter_avg_metrics": m_score}


def validate(dataloader, model, mode: str, loss_fn, metrics_fn, show: bool, show_dir: str =None):
    # model.eval()
    val_loss = []
    val_score = []
    with torch.no_grad():
        for image, sv, mask, edge, meta in tqdm(dataloader):
            target = mask.cuda()
            target_edge = edge.cuda()
            pred = model((image.cuda(), sv.cuda()))
            loss = loss_fn(pred, target, target_edge)
            val_loss.append(loss.detach().item())
            val_score.append(metrics_fn(torch.sigmoid(pred[-1]).cpu(), mask.int()).item())
            if len(val_loss) == 1:
                break
    m_loss = np.mean(val_loss)                         
    m_score = np.mean(val_score)
    print(f"val_data\n loss: {m_loss:.5f}, score: {m_score:.5f}\n")
    
    # plot validation result
    if show:
        num_graghs = len(pred) + 3 # image, mask, edge
        col =  4
        row = math.ceil(num_graghs / col)
        origin_h, origin_w = (meta['height'][0], meta['width'][0])
        plt.figure(figsize=(16, 20))
        for i in range(len(pred)):
            plt.subplot(row, col, i + 1)
            up_sample = F.interpolate(torch.sigmoid(pred[i]), size=(origin_h, origin_w), mode='bilinear', align_corners=False)
            plt.imshow(up_sample.squeeze().cpu().numpy(), cmap='gray')
            plt.axis('off')
            plt.title(f"Output {i}")
            
        plt.subplot(row, col, len(pred) + 1)
        plt.imshow(Image.open(meta['filename_image'][0]).convert("RGB"))
        plt.axis('off')
        plt.title("Image")
        
        plt.subplot(row, col, len(pred) + 2)
        plt.imshow(Image.open(meta['filename_mask'][0]).convert("L"))
        plt.axis('off')
        plt.title("Mask")
        
        # plt.subplot(row, col, len(pred) + 3)
        # plt.imshow(Image.open(meta['filename_edge'][0]).convert("L"))
        # plt.axis('off')
        # plt.title("Edge")
        
        plt.tight_layout()
        plt.savefig(osp.join(show_dir, osp.basename(meta['filename_image'][0])).replace(".png", f"_{mode}.png"))
    
    return {"iter_avg_loss": m_loss, "iter_avg_metrics": m_score}