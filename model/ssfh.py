import math
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.models import resnext101_32x8d

from layers import BasicConv, Basic_TransConv, Relation_Attention, Resudial_Block, Contrast_Module_Deep, CBAM


class Encoder_Decoder(nn.Module):
    def __init__(self, in_c, feature_map_size=416):
        super(Encoder_Decoder, self).__init__()
        
        self.conv0 = BasicConv(in_planes=in_c, out_planes=64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.conv1 = BasicConv(in_planes=64, out_planes=128, kernel_size=3, stride=2, padding=1)
        self.conv2 = BasicConv(in_planes=128, out_planes=256, kernel_size=3, stride=2, padding=1)
        
        self.t_conv1 = Basic_TransConv(in_planes=256, out_planes=128, kernel_size=2, stride=2)
        self.t_conv2 = Basic_TransConv(in_planes=128, out_planes=64, kernel_size=2, stride=2)
        # self.t_conv3 = Basic_TransConv(in_planes=, out_planes=64, kernel_size=2, stride=2)
        
        self.final_conv = nn.Conv2d(13+64, 1, 3, 1, 1)
        
        
    def forward(self, x): 
        # encode
        _ = self.conv0(x)
        _ = self.pool(_)
        _ = self.conv1(_)
        _ = self.conv2(_)
        # decode
        _ = self.t_conv1(_)
        _ = self.t_conv2(_)
        output = F.interpolate(input=_ , size=x.shape[-2:], scale_factor=None, mode='bilinear', align_corners=True)
        output = self.final_conv(torch.cat((x[:, 3:], output), 1))
        return output


class Refinement_Net(nn.Module): #no localrelation
    def __init__(self, in_c):
        super(Refinement_Net, self).__init__()
        self.conv1 = BasicConv(in_planes=in_c, out_planes=64, kernel_size=3, stride=1, padding=1)
        # self.rl1 = LocalRelationalLayer(64)
        self.conv2 = BasicConv(in_planes=64, out_planes=64, kernel_size=3, stride=1, padding=1)
        # self.rl2 = LocalRelationalLayer(128)

        self.res1 = Resudial_Block(64)
        self.res2 = Resudial_Block(64)
        self.res3 = Resudial_Block(64)
        # self.res4 = Resudial_Block(64)

        self.final_conv = nn.Conv2d(13 + 64, 1, 3, 1, 1)

    def forward(self, image, predict_map):
        fusion = torch.cat((image, predict_map), 1)
        
        fusion = self.conv1(fusion)
        fusion = self.conv2(fusion)
        fusion = self.res1(fusion)
        fusion = self.res2(fusion)
        fusion = self.res3(fusion)
        # fusion = self.res4(fusion)
        fusion = self.final_conv(torch.cat((predict_map, fusion), 1))
        return fusion
    
    # ３階層ぐらいでも
    # refineの複雑さが高すぎて、各コンポーネントが十分に学習できない.
    # 空間的な間、レベル感を統合してしまう
    # 同時に学習を始めると、
    # 全体の形状を見た判断ができない　→　値をもっているかどうか
    # rccl, ssf, shが個別に最大限に学習
    # rccl, ssf, shが相互に頑張ることがない　→　refineはその結果をまとめなければならない
    # 全体の範囲を確認できない conv  transconv encoder decoder（）

def INF(B,H,W):
     return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H),0).unsqueeze(0).repeat(B*W,1,1)


class SSF_Extractor(nn.Module):
    def __init__(self, in_channels, patch_size = [3, 6, 8, 10], stride = [1, 2, 3, 4]):
        super(SSF_Extractor, self).__init__()
        
        self.patch_size = 5
        self.p_step = 1
        
    def forward(self, x):
        
        batch, channel, height, width = x.shape
        
        n_ph = (height - self.patch_size) // self.p_step + 1
        n_pw = (width - self.patch_size) // self.p_step + 1
        
        var_attention = torch.zeros(batch, channel, n_ph, n_pw).cuda()
        
        for b in range(batch):
            patch = x[b].unfold(1, self.patch_size, self.p_step).unfold(2, self.patch_size, self.p_step) #(C,nh,nw,ph,pw)
            patch = patch.contiguous().view(channel, n_ph, n_pw, self.patch_size*self.patch_size)
            var_attention[b] = torch.var(patch, dim=-1)

        var_attention = F.interpolate(var_attention, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        return x * var_attention


class SH_Block(nn.Module):
    def __init__(self, planes, pooling_size = None, pool_type = ['max', 'avg']):
        super(SH_Block, self).__init__()

        self.inplanes = int(planes)
        self.inplanes_half = int(planes / 2)
        self.inplanes_quarter= int(planes / 4)
        
        self.k_size = pooling_size
        self.pool_type = pool_type
        
        self.weight = nn.Parameter(torch.tensor([1.]))
        self.bias = nn.Parameter(torch.tensor([0.]))
        
        
    def forward(self, x, sv):

        saturation = (1 - sv[:,0,:,:]).unsqueeze(1) # 0 ~ 1
        value = sv[:,1,:,:].unsqueeze(1)
        
        highlight_map = self.weight * (saturation * value) + self.bias
        highlight_map = torch.sigmoid(highlight_map)
        
        attention_map = 0
        for pool in self.pool_type:
            if pool == 'max':
                attention_map += F.max_pool2d(highlight_map, self.k_size, self.k_size)
            elif pool == 'avg':
                attention_map += F.avg_pool2d(highlight_map, self.k_size, self.k_size)
        
        attention_map = attention_map.repeat(1, self.inplanes, 1, 1)
        
        return x * attention_map


class ZeroOutput(nn.Module):
    def forward(self, x):
        return torch.zeros_like(x)


###################################################################
# ########################## NETWORK ##############################
###################################################################
class SSFH(nn.Module):
    def __init__(self, rccl_zero = False, ssf_zero = False, sh_zero = False, EDF_zero = False, refine_target=False):
        super(SSFH, self).__init__()
        print("\n This model is ssfh.py.")

        self.rccl_flag = rccl_zero
        self.ssf_flag = ssf_zero
        self.sh_flag = sh_zero
        self.edf_flag = EDF_zero
        
        resnext = resnext101_32x8d(pretrained=True, progress=True)
        for param in resnext.parameters():
            param.requires_grad = False
        net_list = list(resnext.children())
        self.layer0 = nn.Sequential(*net_list[:4]) # 64
        self.layer1 = net_list[4] #256
        self.layer2 = net_list[5] #512
        self.layer3 = net_list[6] #1024
        self.layer4 = net_list[7] #2048
        
        '''EDF'''
        self.tmp_edge_extract = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64)) # layer1
        self.tmp_edge_predict = nn.Sequential(nn.Conv2d(64+512*3, 1, 3, 1, 1))

        '''PMDNet'''
        self.rccl_contrast_4 = Contrast_Module_Deep(2048,d1=2, d2=4) # 2048x 12x12
        self.rccl_contrast_3 = Contrast_Module_Deep(1024,d1=4, d2=8) # 1024x 24x24
        self.rccl_contrast_2 = Contrast_Module_Deep(512, d1=4, d2=8) # 512x 48x48
        self.rccl_contrast_1 = Contrast_Module_Deep(256, d1=4, d2=8) # 256x 96x96

        self.rccl_ra_4 = Relation_Attention(2048, 2048)
        self.rccl_ra_3 = Relation_Attention(1024, 1024)
        self.rccl_ra_2 = Relation_Attention(512, 512)
        self.rccl_ra_1 = Relation_Attention(256, 256)

        self.rccl_up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1), nn.BatchNorm2d(512), nn.ReLU())
        self.rccl_up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1), nn.BatchNorm2d(256), nn.ReLU())
        self.rccl_up_2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1), nn.BatchNorm2d(128), nn.ReLU())
        self.rccl_up_1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1), nn.BatchNorm2d(64), nn.ReLU())

        self.rccl_cbam_4 = CBAM(512)
        self.rccl_cbam_3 = CBAM(256)
        self.rccl_cbam_2 = CBAM(128)
        self.rccl_cbam_1 = CBAM(64)

        self.rccl_layer4_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.rccl_layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.rccl_layer2_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.rccl_layer1_predict = nn.Conv2d(64, 1, 3, 1, 1)

        '''SSF'''
        self.ssf_layer4 = SSF_Extractor(in_channels=2048, patch_size=[3, 5], stride=[1, 1])
        self.ssf_layer3 = SSF_Extractor(in_channels=1024, patch_size=[3, 5], stride=[1, 2])
        self.ssf_layer2 = SSF_Extractor(in_channels=512, patch_size=[3, 5], stride=[1, 2])
        self.ssf_layer1 = SSF_Extractor(in_channels=256, patch_size=[3, 5], stride=[1, 2])
                
        self.ssf_up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1),  nn.BatchNorm2d(512), nn.ReLU()) #非線形変換を持たせる
        self.ssf_up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1),  nn.BatchNorm2d(256), nn.ReLU()) # 線形変換だけでは全部同じように変換される
        self.ssf_up_2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1),  nn.BatchNorm2d(128), nn.ReLU()) # CBAM 非線形にチャンネル方向にAttention
        self.ssf_up_1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1),  nn.BatchNorm2d(64), nn.ReLU())
        
        self.ssf_cbam_4 = CBAM(512)
        self.ssf_cbam_3 = CBAM(256)
        self.ssf_cbam_2 = CBAM(128)
        self.ssf_cbam_1 = CBAM(64)
        
        self.ssf_layer4_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.ssf_layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.ssf_layer2_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.ssf_layer1_predict = nn.Conv2d(64, 1, 3, 1, 1)
        
        '''specualr highlight'''
        self.sh_layer4 = SH_Block(planes=2048, pooling_size=32, pool_type=['max'])
        self.sh_layer3 = SH_Block(planes=1024, pooling_size=16, pool_type=['max'])
        self.sh_layer2 = SH_Block(planes=512, pooling_size=8, pool_type=['max'])
        self.sh_layer1 = SH_Block(planes=256, pooling_size=4, pool_type=['max'])

        self.sh_cbam_4 = CBAM(512)
        self.sh_cbam_3 = CBAM(256)
        self.sh_cbam_2 = CBAM(128)
        self.sh_cbam_1 = CBAM(64)

        self.sh_up_4 = nn.Sequential(nn.ConvTranspose2d(2048, 512, 4, 2, 1),  nn.BatchNorm2d(512), nn.ReLU()) 
        self.sh_up_3 = nn.Sequential(nn.ConvTranspose2d(1024, 256, 4, 2, 1),  nn.BatchNorm2d(256), nn.ReLU())
        self.sh_up_2 = nn.Sequential(nn.ConvTranspose2d(512, 128, 4, 2, 1),  nn.BatchNorm2d(128), nn.ReLU()) 
        self.sh_up_1 = nn.Sequential(nn.ConvTranspose2d(256, 64, 4, 2, 1),  nn.BatchNorm2d(64), nn.ReLU())

        self.sh_layer4_predict = nn.Conv2d(512, 1, 3, 1, 1)
        self.sh_layer3_predict = nn.Conv2d(256, 1, 3, 1, 1)
        self.sh_layer2_predict = nn.Conv2d(128, 1, 3, 1, 1)
        self.sh_layer1_predict = nn.Conv2d(64, 1, 3, 1, 1)
        
        # Refinement Network
        if refine_target:
            self.tmp_refinement = Encoder_Decoder(3 + 4 * 3 + 1)
        else:
            self.tmp_refinement =  nn.Conv2d(3+4*3+1, 1, 1, 1, 0)
        
        
        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

    def forward(self, x):
        
        x, sv = x
        
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0) # 256, 104, 104
        layer2 = self.layer2(layer1) # 512, 52, 52
        layer3 = self.layer3(layer2) # 1024, 26, 26
        layer4 = self.layer4(layer3) # 2048, 13, 13

        '''PMD'''
        contrast_4 = self.rccl_contrast_4(layer4)
        cc_att_map_4 = self.rccl_ra_4(layer4)
        final_contrast_4 = contrast_4 * cc_att_map_4

        up_4 = self.rccl_up_4(final_contrast_4)
        cbam_4 = self.rccl_cbam_4(up_4)
        layer4_rccl_predict = self.rccl_layer4_predict(cbam_4)
        layer4_map = torch.sigmoid(layer4_rccl_predict)

        contrast_3 = self.rccl_contrast_3(layer3 * layer4_map)
        cc_att_map_3 = self.rccl_ra_3(layer3 * layer4_map)

        final_contrast_3 = contrast_3 * cc_att_map_3

        up_3 = self.rccl_up_3(final_contrast_3)
        cbam_3 = self.rccl_cbam_3(up_3)
        layer3_rccl_predict = self.rccl_layer3_predict(cbam_3)
        layer3_map = torch.sigmoid(layer3_rccl_predict)

        contrast_2 = self.rccl_contrast_2(layer2 * layer3_map)
        cc_att_map_2 = self.rccl_ra_2(layer2 * layer3_map)
        final_contrast_2 = contrast_2 * cc_att_map_2

        up_2 = self.rccl_up_2(final_contrast_2)
        cbam_2 = self.rccl_cbam_2(up_2)
        layer2_rccl_predict = self.rccl_layer2_predict(cbam_2)
        layer2_map = torch.sigmoid(layer2_rccl_predict)

        contrast_1 = self.rccl_contrast_1(layer1 * layer2_map)
        cc_att_map_1 = self.rccl_ra_1(layer1 * layer2_map)
        final_contrast_1 = contrast_1 * cc_att_map_1

        up_1 = self.rccl_up_1(final_contrast_1)
        cbam_1 = self.rccl_cbam_1(up_1)
        layer1_rccl_predict = self.rccl_layer1_predict(cbam_1)

        '''SSF'''
        ssf_4 = self.ssf_layer4(layer4)
        ssf_up4 = self.ssf_up_4(ssf_4)
        ssf_cbam4 = self.ssf_cbam_4(ssf_up4)
        layer4_ssf_predict = self.ssf_layer4_predict(ssf_cbam4)
        layer4_ssf_map = torch.sigmoid(layer4_ssf_predict)
        
        ssf_3 = self.ssf_layer3(layer3)
        ssf_up3 = self.ssf_up_3(ssf_3 * layer4_ssf_map)
        ssf_cbam3 = self.ssf_cbam_3(ssf_up3)
        layer3_ssf_predict = self.ssf_layer3_predict(ssf_cbam3)
        layer3_ssf_map = torch.sigmoid(layer3_ssf_predict)
        
        ssf_2 = self.ssf_layer2(layer2)
        ssf_up2 = self.ssf_up_2(ssf_2 * layer3_ssf_map)
        ssf_cbam2 = self.ssf_cbam_2(ssf_up2)
        layer2_ssf_predict = self.ssf_layer2_predict(ssf_cbam2)
        layer2_ssf_map = torch.sigmoid(layer2_ssf_predict)
        
        ssf_1 = self.ssf_layer1(layer1)
        ssf_up1 = self.ssf_up_1(ssf_1 * layer2_ssf_map)
        ssf_cbam1 = self.ssf_cbam_1(ssf_up1)
        layer1_ssf_predict = self.ssf_layer1_predict(ssf_cbam1)
        
        ''' Specualr Highlight'''
        highlight_4 = self.sh_layer4(layer4, sv)
        highlight_4_up = self.sh_up_4(highlight_4)
        sh_cbam_4 = self.sh_cbam_4(highlight_4_up)
        sh4_predict = self.sh_layer4_predict(sh_cbam_4)
        sh4_map = torch.sigmoid(sh4_predict)

        highlight_3 = self.sh_layer3(layer3, sv)
        highlight_3_up = self.sh_up_3(highlight_3 * sh4_map)
        sh_cbam_3 = self.sh_cbam_3(highlight_3_up)
        sh3_predict = self.sh_layer3_predict(sh_cbam_3)
        sh3_map = torch.sigmoid(sh3_predict)

        highlight_2 = self.sh_layer2(layer2, sv)
        highlight_2_up = self.sh_up_2(highlight_2 * sh3_map)
        sh_cbam_2 = self.sh_cbam_2(highlight_2_up)
        sh2_predict = self.sh_layer2_predict(sh_cbam_2)
        sh2_map = torch.sigmoid(sh2_predict)

        highlight_1 = self.sh_layer1(layer1, sv)
        highlight_1_up = self.sh_up_1(highlight_1 * sh2_map)
        sh_cbam_1 = self.sh_cbam_1(highlight_1_up)
        sh1_predict = self.sh_layer1_predict(sh_cbam_1)
        
        '''Intermidiate Mirror Map'''
        
        layer4_rccl_predict = F.interpolate(layer4_rccl_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
        layer3_rccl_predict = F.interpolate(layer3_rccl_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
        layer2_rccl_predict = F.interpolate(layer2_rccl_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
        layer1_rccl_predict = F.interpolate(layer1_rccl_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
        
        '''Intermidiate SSF Map'''
        layer4_ssf_predict = F.interpolate(layer4_ssf_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_ssf_predict = F.interpolate(layer3_ssf_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
        layer2_ssf_predict = F.interpolate(layer2_ssf_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
        layer1_ssf_predict = F.interpolate(layer1_ssf_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
        
        '''Intermidiate SH Map'''
        sh4_predict = F.interpolate(sh4_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        sh3_predict = F.interpolate(sh3_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        sh2_predict = F.interpolate(sh2_predict, size=x.size()[2:], mode='bilinear', align_corners=True)
        sh1_predict = F.interpolate(sh1_predict, size=x.size()[2:], mode='bilinear', align_corners=True) 
    
        # 個別の特徴抽出器を学習させる目的
        if self.rccl_flag:
            layer4_rccl_predict = self.output_zero(layer4_rccl_predict)
            layer3_rccl_predict = self.output_zero(layer3_rccl_predict)
            layer2_rccl_predict = self.output_zero(layer2_rccl_predict)
            layer1_rccl_predict = self.output_zero(layer1_rccl_predict)
            cbam_4 = self.output_zero(cbam_4)

        if self.ssf_flag:
            layer4_ssf_predict = self.output_zero(layer4_ssf_predict)
            layer3_ssf_predict = self.output_zero(layer3_ssf_predict)
            layer2_ssf_predict = self.output_zero(layer2_ssf_predict)
            layer1_ssf_predict = self.output_zero(layer1_ssf_predict)
            ssf_cbam4 = self.output_zero(ssf_cbam4)

        if self.sh_flag:
            sh4_predict = self.output_zero(sh4_predict) 
            sh3_predict = self.output_zero(sh3_predict) 
            sh2_predict = self.output_zero(sh2_predict) 
            sh1_predict = self.output_zero(sh1_predict) 
            sh_cbam_4 = self.output_zero(sh_cbam_4) 

        '''Edge'''
        edge_feature = self.tmp_edge_extract(layer1)
        layer4_features = torch.cat((cbam_4, ssf_cbam4, sh_cbam_4), 1)
        layer4_edge_feature = F.interpolate(layer4_features, size=edge_feature.size()[2:], mode='bilinear', align_corners=True)
        final_edge_feature = torch.cat((edge_feature, layer4_edge_feature), 1)
        layer0_edge = self.tmp_edge_predict(final_edge_feature)
        if self.edf_flag:
            layer0_edge = self.output_zero(layer0_edge)
        layer0_edge = F.interpolate(layer0_edge, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        '''Final Mirror Map'''
        final_features = torch.cat((layer0_edge, layer4_rccl_predict, layer3_rccl_predict, layer2_rccl_predict, layer1_rccl_predict, layer4_ssf_predict, layer3_ssf_predict, layer2_ssf_predict, layer1_ssf_predict, sh4_predict, sh3_predict, sh2_predict, sh1_predict), 1) 
        # final_predict = self.tmp_refinement(x, final_features)
        final_predict = self.tmp_refinement(torch.cat((x, final_features), 1))
        

        if self.training == True:
            return layer4_rccl_predict, layer3_rccl_predict, layer2_rccl_predict, layer1_rccl_predict, layer4_ssf_predict, layer3_ssf_predict, layer2_ssf_predict, layer1_ssf_predict, sh4_predict, sh3_predict, sh2_predict, sh1_predict, layer0_edge, final_predict
        
        return torch.sigmoid(layer4_rccl_predict), torch.sigmoid(layer3_rccl_predict), torch.sigmoid(layer2_rccl_predict), torch.sigmoid(layer1_rccl_predict), torch.sigmoid(layer4_ssf_predict), torch.sigmoid(layer3_ssf_predict), torch.sigmoid(layer2_ssf_predict), torch.sigmoid(layer1_ssf_predict), torch.sigmoid(sh4_predict), torch.sigmoid(sh3_predict), torch.sigmoid(sh2_predict), torch.sigmoid(sh1_predict), torch.sigmoid(layer0_edge), torch.sigmoid(final_predict)
        
    def reset_tmp_parameters(self):
        self.tmp_edge_extract = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64)) # layer1
        self.tmp_edge_predict = nn.Sequential(nn.Conv2d(64+512*3, 1, 3, 1, 1))
        self.tmp_refinement = Refinement_Net(3+1+4*3)
        self.tmp_edge_extract = self.tmp_edge_extract.cuda()
        self.tmp_edge_predict = self.tmp_edge_predict.cuda()
        self.tmp_refinement = self.tmp_refinement.cuda()
    
    def output_zero(self, x):
        return x * torch.zeros_like(x)


