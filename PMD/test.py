import pdb
import os

import torch
import torch.nn as nn
from tqdm import tqdm
import numpy as np
from matplotlib import pyplot as plt
from torchvision.transforms import Resize

from metrics.fmeasure import calculate_max_fbeta
from sklearn.metrics import mean_absolute_error, jaccard_score


def test(dataloder, model, save_dir: str):
    model.eval()
    maxF_dir = os.path.join(save_dir, "maxFscore")
    os.makedirs(maxF_dir, exist_ok=True)
    
    test_iou = []
    test_fbeta = []
    test_mae = []
    
    with torch.no_grad():
        for image, sv, mask, edge, meta in tqdm(dataloder):
            input = (image.cuda(), sv.cuda())
            pred = model(input)
            # loss = loss_fn(pred, target, target_edge)
            # test_loss.append(loss.detach().item())
            # test_score.append(metrics_fn(torch.sigmoid(pred[-1]).cpu(), mask.int()).item())
            upsample = nn.Upsample((meta['height'], meta['width']), mode='bilinear', align_corners=True)

            up_pred = upsample(pred[-1].cpu())
            up_mask = upsample(mask.cpu())
            
            pred1d = up_pred.numpy().flatten()
            true1d = up_mask.numpy().astype(np.uint8).flatten()
            maxFscore, thres = calculate_max_fbeta(true1d, pred1d)
            mae = mean_absolute_error(true1d, pred1d)
            iou = jaccard_score(true1d, pred1d > 0.5)
            
            # print
            print("filename: ", meta['filename_image'][0])
            print("maxFscore: ", maxFscore, "threshold: ", thres)
            print("MAE: ", mae)
            print("IoU: ", iou)
            
            test_fbeta.append(maxFscore)
            test_mae.append(mae)
            test_iou.append(iou)
            
            # save predict image
            up_pred[up_pred > thres] = 255
            up_pred[up_pred <= thres] = 0
            filename = os.path.basename(meta['filename_mask'][0])
            plt.imsave(os.path.join(maxF_dir, filename), up_pred.squeeze().numpy().astype(np.uint8), cmap='gray')
            

    # Average
    m_maxFscore = np.mean(test_fbeta)
    m_mae = np.mean(test_mae)
    m_iou = np.mean(test_iou)

    print(f"test_data\n maxFscore: {m_maxFscore:.5f}, MAE: {m_mae:.5f}, IoU: {m_iou:.5f}\n")
    
    return {"maxFscore": m_maxFscore, "MAE": m_mae, "IoU": m_iou}
