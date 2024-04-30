import random
import pdb
import os
import os.path as osp
import time
from argparse import ArgumentParser

import torch
from torch import optim
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
from torchvision import transforms
from torchmetrics.classification import BinaryJaccardIndex
from torchmetrics.classification import BinaryFBetaScore
from torch.utils.data import ConcatDataset
import matplotlib.pyplot as plt
from datetime import datetime as dt

from loss.loss import LossComponent, Loss_Refine_EDF
from model.network_sub import Network
import earlystop
from train import train, validate
from test import test
from learning_component import model_param_reading, model_setting, load_weights_partial
from pmd_config import make_PMD_loader

def parse_args():
    '''コマンドライン引数'''
    parser = ArgumentParser()
    parser.add_argument('-dataset_path', type=str, default="data/plastic_mold_dataset",help="dataset path") 
    parser.add_argument('-mode', type=str, choices=['rccl', 'ssf', 'sh', 'refine', 'pmd'], required=True)
    parser.add_argument('-write_dir', type=str, required=True)
    parser.add_argument('-seed', type=int, default=0, help="random seed")
    
    parser.add_argument('--train', action="store_true", help="learning model.")
    
    parser.add_argument('--eval', action="store_true", help="predict test data.")
    parser.add_argument('-epochs', type=int, default=150, help='defalut:150')
    parser.add_argument('-batch_size', type=int, default=5, help='Defalut:5')
    parser.add_argument('-resize', type=int, default=416 , help='Default:416')
    parser.add_argument("-patient", type=int, default=10, help="Early Stopping . the number of epoch. defalut 10")
    # tmp
    parser.add_argument('-final_loss_weight', type=int, default=4, help="Refinement Loss weight.")
    parser.add_argument('-read_weight_path', type=str, default=None)

    return parser.parse_args()



def main():
    opt = parse_args()

    """ dataset """
    train_loader, val_loader, test_loader = make_PMD_loader(opt.dataset_path, seed=opt.seed, batch_size=opt.batch_size, scale=opt.resize)

    """ directory structure """
    result_root_check = os.path.join(opt.write_dir, "checkpoint")
    result_root_output = os.path.join(opt.write_dir, "output")
    result_root_check_output = os.path.join(opt.write_dir, "validation_predict")
    result_root_plot = os.path.join(opt.write_dir, "loss_metrics_plot")
    component_param_path = os.path.join(result_root_check, f"{opt.mode}.pth")
    
    # Make directories
    os.makedirs(opt.write_dir, exist_ok=True)
    os.makedirs(result_root_check, exist_ok=True)
    os.makedirs(result_root_output, exist_ok=True)
    os.makedirs(result_root_check_output, exist_ok=True)
    os.makedirs(result_root_plot, exist_ok=True)


    """ setting model"""
    if opt.mode == "rccl":
        model = Network(rccl_learn=True).cuda()
        
        new_state_dict = {}
        param = torch.load("/data2/yoshimura/mirror_detection/specularflow_highlight_deepNet/PMD/pmd.pth")
        param2 = model.rccl_net.state_dict()
        for key, value in param.items():
            # print(key)
            if 'edge' in key:
                continue
            elif 'layer4_predict' in key:
                new_key = key.replace('layer4_predict', 'rccl_layer4_predict')
                
            elif 'layer3_predict' in key:
                new_key = key.replace('layer3_predict', 'rccl_layer3_predict')
            elif 'layer2_predict' in key:
                new_key = key.replace('layer2_predict', 'rccl_layer2_predict')
            elif 'layer1_predict' in key:
                new_key = key.replace('layer1_predict', 'rccl_layer1_predict')
            
            elif 'layer' in key:
                continue
            elif 'refinement' in key:
                continue
            else:
                new_key = 'rccl_' + key
            new_state_dict[new_key] = value
            print(f"{key} -> {new_key}")
        
        model.rccl_net.load_state_dict(new_state_dict, strict=False)
        model = model_setting(model, rccl_freeze=False, ssf_freeze=True, sh_freeze=True, EDF_freeze=True)

    elif opt.mode == "ssf":
        model = Network(ssf_learn=True).cuda()
        model = model_setting(model, rccl_freeze=True, ssf_freeze=False, sh_freeze=True, EDF_freeze=True)

    elif opt.mode == "sh":
        model = Network(sh_learn=True).cuda()
        model = model_setting(model, rccl_freeze=True, ssf_freeze=True, sh_freeze=False, EDF_freeze=True)

    elif opt.mode == "refine":
        model = Network().cuda()
        # Load parameters 
        rccl_param_path = osp.join(result_root_check, "rccl.pth") 
        ssf_param_path = osp.join(result_root_check, "ssf.pth") 
        sh_param_path = osp.join(result_root_check, "sh.pth") 
        model = load_weights_partial(model, rccl_param_path, 'rccl')
        model = load_weights_partial(model, ssf_param_path, 'ssf')
        model = load_weights_partial(model, sh_param_path, 'sh')
        # Freeze
        model = model_setting(model, rccl_freeze=True, ssf_freeze=True, sh_freeze=True, EDF_freeze=False)

    elif opt.mode == "pmd":
        model = Network(pmd_learn=True).cuda()
        model = model_param_reading(model, os.path.join(result_root_check, "rccl.pth"), ["rccl"])
        model = model_setting(model, rccl_freeze=True, ssf_freeze=True, sh_freeze=True, EDF_freeze=False)

    else:
        print("No such component.")


    """ Learning phase """
    if opt.train:
        if (opt.mode == "refine") or (opt.mode == "pmd"):
            loss_fn = Loss_Refine_EDF(W_edge=5, W_final=1)
        else:
            loss_fn = LossComponent(W_map=1, W_final=1)
        metrics_fn = BinaryFBetaScore(beta=0.5)
        es = earlystop.EarlyStopping(
                                    verbose=True,
                                    patience=opt.patient, 
                                    path=component_param_path
                                    )
        # optimizer
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        train_epoch_loss = []
        train_epoch_metrics = []
        val_epoch_loss = []
        val_epoch_metrics = []
        print("model training... ")
        for epoch in range(opt.epochs):
            print('\nEpoch: {}'.format(epoch+1))
            # training
            train_eval_dict = train(train_loader, model, loss_fn, metrics_fn, optimizer)
            train_epoch_loss.append(train_eval_dict["iter_avg_loss"])
            train_epoch_metrics.append(train_eval_dict["iter_avg_metrics"])
            # validation
            val_eval_dict = validate(val_loader, model, opt.mode, loss_fn, metrics_fn, show=True, show_dir=result_root_check_output)
            val_epoch_loss.append(val_eval_dict["iter_avg_loss"])
            val_epoch_metrics.append(val_eval_dict["iter_avg_metrics"])
            
            # es(v_loss, model)
            es(val_eval_dict["iter_avg_loss"], model)
            if es.early_stop:
                print("Early Stopping.")
                break
            
            # save loss graghs
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].plot(train_epoch_loss, label="train_loss")
            axes[0].plot(val_epoch_loss, label="val_loss")
            axes[0].legend()
            axes[1].plot(train_epoch_metrics, label="train_metrics")
            axes[1].plot(val_epoch_metrics, label="val_metrics")
            axes[1].legend()
            plt.savefig(os.path.join(result_root_plot, f"loss_metrics_{opt.mode}.png"))
            plt.close()
            
    """ Testing phase"""
    if opt.eval:
        print("Test phase:")
        print("Read parameter")
        test_loader.dataset.train_mode = False
        if not os.path.exists(component_param_path):
            raise FileNotFoundError(f"Not found path: {component_param_path}")

        pred_dir = os.path.join(result_root_output, opt.mode)
        os.makedirs(pred_dir, exist_ok=True)
        # Predict step and evaluation scores
        eval_dict = test(test_loader, model, pred_dir)

        with open(os.path.join(opt.write_dir, "eval.txt"), mode="a") as w: 
            w.write(f"{opt.mode}:\n")
            w.write("Fbeta: {:.5f}, MAE: {:.5f}, IoU: {:.5f}".format(eval_dict['maxFscore'], eval_dict['MAE'], eval_dict['IoU']))
    else:
        print("No learn and evaluate.")
    print("finished!")



if __name__ == '__main__':
    main()