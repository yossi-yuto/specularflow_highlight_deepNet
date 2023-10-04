import random
import pdb
import os
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
from tqdm import tqdm

from loss.dice_bce import DBCE
from model.network import Network
from dataset import SSF_SH_Dataset
from asset import visialize
import earlystop
from learning_component import train, val, test, plot_loss_gragh, model_param_reading, model_setting, load_rccl_ssf_sh_and_refine_params


def parse_args():
    '''コマンドライン引数'''
    parser = ArgumentParser()
    parser.add_argument('-dataset_path', type=str, default="data/plastic_mold_dataset",help="dataset path") 
    parser.add_argument('-test_mold_type', type=str, default=None, help="except mold kind.")
    parser.add_argument('-mode', type=str, choices=['rccl', 'ssf', 'sh', 'refine', 'pmd'], required=True)
    parser.add_argument('-write_dir', type=str, required=True)
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


# type cross validation
def mold_dataset(args):
    assert os.path.exists(args.dataset_path), f"Not found {args.dataset_path}"
    assert args.test_mold_type != None, "No setting test mold type."
    print("mold dataset make:\n")
    train_dataset = []
    val_dataset = []
    for mold_type in os.listdir(args.dataset_path):
        mold_type_dir = os.path.join(args.dataset_path, mold_type)
        img_dir = os.path.join(mold_type_dir, "image")
        mask_dir = os.path.join(mold_type_dir, "mask")
        imgs_path = [os.path.join(img_dir, p) for p in os.listdir(img_dir)]

        if args.test_mold_type == mold_type:
            test_imgs = imgs_path
            test_mask_dir = mask_dir
            print(f"{mold_type} <- test: {len(test_imgs)}")
        else:
            dataset = SSF_SH_Dataset(imgs_path, mask_dir)
            train_size, val_size = getTrainTestCounts(dataset)
            t_dataset, v_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
            train_dataset.append(t_dataset)
            val_dataset.append(v_dataset)
            print(f"{mold_type} <- learning, train:{train_size}, val:{val_size}")
            
    train_dataset = ConcatDataset(train_dataset)
    val_dataset = ConcatDataset(val_dataset)
    print(f"train: {len(train_dataset)}, valid: {len(val_dataset)}")

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1, num_workers=0, pin_memory=True)
    
    return  train_loader, val_loader, test_imgs, test_mask_dir


# spherical_mirror_dataset
def spherical_mirror_dataset(args):
    assert os.path.exists(args.dataset_path), f"Not found {args.dataset_path}"
    train_path = os.path.join(args.dataset_path, "train")
    test_path = os.path.join(args.dataset_path, "test")
    train_img_path = os.path.join(train_path, "image")
    train_mask_path = os.path.join(train_path, "mask")
    test_img_path = os.path.join(test_path, "image")
    test_mask_path = os.path.join(test_path, "mask")
    # image list 
    train_imgs = [os.path.join(train_img_path, f) for f in os.listdir(train_img_path)]
    test_imgs = [os.path.join(test_img_path, f) for f in os.listdir(test_img_path)]
    # dataset 
    dataset = SSF_SH_Dataset(train_imgs, train_mask_path)
    train_size, val_size = getTrainTestCounts(dataset)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
    print(f" train: {train_dataset.__len__()},\n val: {val_dataset.__len__()},\n test: {len(test_imgs)}")
    train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True,num_workers=0, pin_memory= True)
    val_loader = torch.utils.data.DataLoader(val_dataset,batch_size=1,num_workers=0, pin_memory= True)
    return train_loader, val_loader, test_imgs, test_mask_path


def getTrainTestCounts(dataset):
    train_size = int(dataset.__len__() * 0.8) 
    val_size   = dataset.__len__() - train_size
    return train_size, val_size


def check_weight_path(result_dir, read_weight_path):
    result_root_check = os.path.join(result_dir, read_weight_path, "checkpoint")
    if not os.path.exists(result_root_check):
        raise FileNotFoundError(f"Not found file {result_root_check}")
    return result_root_check


def main():
    opt = parse_args()

    """ dataset """
    if "mold" in opt.dataset_path:
        result_dir = "result/plastic_mold"
        train_loader, val_loader, test_imgs, test_mask_dir = mold_dataset(opt)
        result_root = os.path.join(result_dir, opt.write_dir)
        print("Plastic mold dataset")
    elif "spherical" in opt.dataset_path:
        train_loader, val_loader, test_imgs, test_mask_dir = spherical_mirror_dataset(opt)
        result_root = os.path.join("result/spherical_mirror", opt.write_dir)
        print("Spherical mirror dataset")
    else:
        print("Not found dataset.")

    """ directory structure """
    result_root_check = os.path.join(result_root, "checkpoint")
    result_root_output = os.path.join(result_root, "output")
    result_root_check_output = os.path.join(result_root, "validation_predict")
    result_root_plot = os.path.join(result_root, "loss_metrics_plot")
    component_param_path = os.path.join(result_root_check, f"{opt.mode}.pth")
    
    # Make directories
    os.makedirs(result_root, exist_ok=True)
    os.makedirs(result_root_check, exist_ok=True)
    os.makedirs(result_root_output, exist_ok=True)
    os.makedirs(result_root_check_output, exist_ok=True)
    os.makedirs(result_root_plot, exist_ok=True)

    """ setting model"""
    if opt.mode == "rccl":
        model = Network(rccl_zero=False, ssf_zero=True, sh_zero=True, EDF_zero=True).cuda()
        model = model_setting(model, rccl_freeze=False, ssf_freeze=True, sh_freeze=True, EDF_freeze=True)

    elif opt.mode == "ssf":
        model = Network(rccl_zero=True, ssf_zero=False, sh_zero=True, EDF_zero=True).cuda()
        model = model_setting(model, rccl_freeze=True, ssf_freeze=False, sh_freeze=True, EDF_freeze=True)

    elif opt.mode == "sh":
        model = Network(rccl_zero=True, ssf_zero=True, sh_zero=False, EDF_zero=True).cuda()
        model = model_setting(model, rccl_freeze=True, ssf_freeze=True, sh_freeze=False, EDF_freeze=True)

    elif opt.mode == "refine":
        model = Network(rccl_zero=False, ssf_zero=False, sh_zero=False, EDF_zero=False).cuda()
        if opt.read_weight_path:
            result_root_check = check_weight_path(result_dir, opt.read_weight_path)
        # model = model_param_reading(model, os.path.join(result_root_check, "rccl.pth"), ["rccl"])
        # model = model_param_reading(model, os.path.join(result_root_check, "ssf.pth"), ["ssf"])
        # model = model_param_reading(model, os.path.join(result_root_check, "sh.pth"), ["sh"])
        
        rccl_param_path = os.path.join(result_root_check, "rccl.pth") 
        ssf_param_path = os.path.join(result_root_check, "ssf.pth") 
        sh_param_path = os.path.join(result_root_check, "sh.pth") 
        model = load_rccl_ssf_sh_and_refine_params(model, rccl_param_path, ssf_param_path, sh_param_path)
        model = model_setting(model, rccl_freeze=True, ssf_freeze=True, sh_freeze=True, EDF_freeze=False)

    elif opt.mode == "pmd":
        model = Network(rccl_zero=False, ssf_zero=True, sh_zero=True, EDF_zero=False).cuda()
        if opt.read_weight_path:
            result_root_check = check_weight_path(result_dir, opt.read_weight_path)
        model = model_param_reading(model, os.path.join(result_root_check, "rccl.pth"), ["rccl"])
        model = model_setting(model, rccl_freeze=True, ssf_freeze=True, sh_freeze=True, EDF_freeze=False)

    else:
        print("No such component.")
        
    

    """ Learning phase """
    if opt.train:
        if opt.mode == "refine":
            loss_fn = DBCE(W_s=0, W_b=5, W_f=1)
        else:
            loss_fn = DBCE(W_s=1, W_b=0, W_f=1)
        metrics_fn = BinaryFBetaScore(beta=0.5)
        es = earlystop.EarlyStopping(
                                    verbose=True,
                                    patience=opt.patient, 
                                    path=component_param_path
                                    )
        # optimizer
        optimizer = optim.Adam(model.parameters())
        train_loss = []
        train_metrics = []
        val_loss = []
        val_metrics = []
        print("model training... ")
        for epoch in range(opt.epochs):
            print('\nEpoch: {}'.format(epoch+1))
            train_output = train(train_loader, 
                                            model, 
                                            loss_fn, 
                                            metrics_fn, 
                                            optimizer
                                            )
            val_output = val(val_loader, 
                                            model, 
                                            loss_fn, 
                                            metrics_fn, 
                                            os.path.join(result_root_check_output, f"{opt.mode}.png"))
            train_loss.append(train_output["mean_refine_loss"])
            train_metrics.append(train_output["mean_metrics"])
            val_loss.append(val_output["mean_refine_loss"])
            val_metrics.append(val_output["mean_metrics"])
            if epoch % 5 == 0:
                plot_loss_gragh(train_loss, val_loss, train_metrics, val_metrics, 
                                os.path.join(result_root_plot, f"{opt.mode}.png"))
            # es(v_loss, model)
            es(val_output["mean_refine_loss"], model)
            if es.early_stop:
                print("Early Stopping.")
                break
            
    """ Testing phase"""
    if opt.eval:
        print("Test phase:")
        print("Read parameter")
        if not os.path.exists(component_param_path):
            raise FileNotFoundError(f"Not found path: {component_param_path}")

        pred_dir = os.path.join(result_root_output, opt.mode)
        os.makedirs(pred_dir, exist_ok=True)
        # Predict step and evaluation scores
        Fbeta, MAE =  test(test_imgs, test_mask_dir, model, pred_dir, component_param_path)

        with open(os.path.join(result_root, "eval.txt"), mode="a") as w: 
            w.write(f"{opt.mode}:\n")
            w.write(f"Fbeta: {Fbeta}, MAE: {MAE}\n")
    else:
        print("No learn and evaluate.")
    print("finished!")



if __name__ == '__main__':
    main()