import math
import pdb

import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torchvision.models import resnext101_32x8d

from .layers import BasicConv, Basic_TransConv, Relation_Attention, Resudial_Block, Contrast_Module_Deep, CBAM


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


class RCCL_SubNet(nn.Module):
    def __init__(self, origin_size=(416, 416)):
        super(RCCL_SubNet, self).__init__()
        
        self.origin_size = origin_size
        
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
        
        self.rccl_refine = nn.Conv2d(4, 1, 1, 1, 0)
        
    def forward(self, feat_maps):
        layer4, layer3, layer2, layer1 = feat_maps
        
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
        
        '''Intermidiate Mirror Map'''
        layer4_rccl_predict = F.interpolate(layer4_rccl_predict, size=self.origin_size, mode='bilinear', align_corners=True) 
        layer3_rccl_predict = F.interpolate(layer3_rccl_predict, size=self.origin_size, mode='bilinear', align_corners=True) 
        layer2_rccl_predict = F.interpolate(layer2_rccl_predict, size=self.origin_size, mode='bilinear', align_corners=True) 
        layer1_rccl_predict = F.interpolate(layer1_rccl_predict, size=self.origin_size, mode='bilinear', align_corners=True) 
        
        rccl_predicts = torch.cat([layer4_rccl_predict, layer3_rccl_predict, layer2_rccl_predict, layer1_rccl_predict], 1)
        rccl_refine = self.rccl_refine(rccl_predicts)
        
        return layer4_rccl_predict, layer3_rccl_predict, layer2_rccl_predict, layer1_rccl_predict, rccl_refine, cbam_4
        
        
class SSF_SubNet(nn.Module):
    def __init__(self, origin_size=(416,416)):
        super(SSF_SubNet, self).__init__()
        
        self.origin_size = origin_size
        
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
        
        self.ssf_refine = nn.Conv2d(4, 1, 1, 1, 0)
        
    def forward(self, feat_maps):
        layer4, layer3, layer2, layer1 = feat_maps
        
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
        
        layer4_ssf_predict = F.interpolate(layer4_ssf_predict, size=self.origin_size, mode='bilinear', align_corners=True)
        layer3_ssf_predict = F.interpolate(layer3_ssf_predict, size=self.origin_size, mode='bilinear', align_corners=True) 
        layer2_ssf_predict = F.interpolate(layer2_ssf_predict, size=self.origin_size, mode='bilinear', align_corners=True) 
        layer1_ssf_predict = F.interpolate(layer1_ssf_predict, size=self.origin_size, mode='bilinear', align_corners=True)
        
        ssf_predicts = torch.cat([layer4_ssf_predict, layer3_ssf_predict, layer2_ssf_predict, layer1_ssf_predict], 1)
        ssf_refine = self.ssf_refine(ssf_predicts)
        
        return layer4_ssf_predict, layer3_ssf_predict, layer2_ssf_predict, layer1_ssf_predict, ssf_refine, ssf_cbam4 
        
        
class SH_SubNet(nn.Module):
    def __init__(self, origin_size=(416, 416)):
        super(SH_SubNet, self).__init__()
        
        self.origin_size = origin_size
        
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
        
        self.sh_refine = nn.Conv2d(4, 1, 1, 1, 0)
        
    def forward(self, feat_maps, sv):
        layer4, layer3, layer2, layer1 = feat_maps
    
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
        
        '''Intermidiate SH Map'''
        sh4_predict = F.interpolate(sh4_predict, size=self.origin_size, mode='bilinear', align_corners=True)
        sh3_predict = F.interpolate(sh3_predict, size=self.origin_size, mode='bilinear', align_corners=True)
        sh2_predict = F.interpolate(sh2_predict, size=self.origin_size, mode='bilinear', align_corners=True)
        sh1_predict = F.interpolate(sh1_predict, size=self.origin_size, mode='bilinear', align_corners=True)
        sh_predicts = torch.cat([sh4_predict, sh3_predict, sh2_predict, sh1_predict], 1)
        sh_refine = self.sh_refine(sh_predicts)
        
        return sh4_predict, sh3_predict, sh2_predict, sh1_predict, sh_refine, sh_cbam_4 
    
    
class EDF_module(nn.Module):
    def __init__(self):
        super(EDF_module, self).__init__()
        self.edge_extract = nn.Sequential(nn.Conv2d(256, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.Conv2d(128, 64, 3, 1, 1), nn.BatchNorm2d(64)) 
        self.edge_predict = nn.Sequential(nn.Conv2d(64+512*3, 1, 3, 1, 1))

    def forward(self, row_layer, high_layers):
        
        low_feature = self.edge_extract(row_layer)
        high_feature = F.interpolate(high_layers, size=low_feature.size()[2:], mode='bilinear', align_corners=True)
        edge_feature = torch.cat([low_feature, high_feature], 1)        
        return self.edge_predict(edge_feature)
        

###################################################################
# ########################## NETWORK ##############################
###################################################################
class Network(nn.Module):
    def __init__(self, rccl_learn = False, ssf_learn = False, sh_learn = False, pmd_learn = False):
        super(Network, self).__init__()

        self.rccl_learn = rccl_learn
        self.ssf_learn = ssf_learn
        self.sh_learn = sh_learn
        self.pmd_learn = pmd_learn
        
        resnext = resnext101_32x8d(pretrained=True, progress=True)
        for param in resnext.parameters():
            param.requires_grad = False
        net_list = list(resnext.children())
        self.layer0 = nn.Sequential(*net_list[:4]) # 64
        self.layer1 = net_list[4] #256
        self.layer2 = net_list[5] #512
        self.layer3 = net_list[6] #1024
        self.layer4 = net_list[7] #2048
        
        # Sub Network 
        self.rccl_net = RCCL_SubNet()
        self.ssf_net = SSF_SubNet()
        self.sh_net = SH_SubNet()
        
        """ EDF """
        self.edf_net = EDF_module()
        
        """ refine """
        self.pmd_refine = nn.Conv2d(2, 1, 1, 1, 0)
        self.refine = nn.Conv2d(4, 1, 1, 1, 0)
        
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
        
        feat_maps = (layer4, layer3, layer2, layer1)

        ''' Sub Network '''
        # RCCL network
        if self.rccl_learn:
            rccl_4, rccl_3, rccl_2, rccl_1, rccl_refine, _ = self.rccl_net(feat_maps)
            return rccl_4, rccl_3, rccl_2, rccl_1, rccl_refine
        else:
            rccl_4, rccl_3, rccl_2, rccl_1, rccl_refine, rccl_cbam = self.rccl_net(feat_maps)
        
        # PMDNet
        if self.pmd_learn:
            high_feature = torch.cat([rccl_cbam, self.output_zero(rccl_cbam), self.output_zero(rccl_cbam)], 1)
            boundary = self.edf_net(layer1, high_feature)
            boundary = F.interpolate(boundary, size=x.size()[2:], mode='bilinear', align_corners=True)
            final_predict = self.refine(torch.cat([torch.sigmoid(rccl_refine), torch.sigmoid(boundary)], 1))
            if self.training:
                return boundary, final_predict
            # test mode
            return torch.sigmoid(boundary), torch.sigmoid(final_predict)
        
        # SSF network
        if self.ssf_learn:
            ssf_4, ssf_3, ssf_2, ssf_1, ssf_refine, _ = self.ssf_net(feat_maps)
            return ssf_4, ssf_3, ssf_2, ssf_1, ssf_refine
        else:
            ssf_4, ssf_3, ssf_2, ssf_1, ssf_refine, ssf_cbam = self.ssf_net(feat_maps)
        # SH network
        if self.sh_learn:
            sh_4, sh_3, sh_2, sh_1, sh_refine, _ = self.sh_net(feat_maps, sv)
            return sh_4, sh_3, sh_2, sh_1, sh_refine
        else:
            sh_4, sh_3, sh_2, sh_1, sh_refine, sh_cbam = self.sh_net(feat_maps, sv)
            
        '''Edge'''
        high_feature = torch.cat([rccl_cbam, ssf_cbam, sh_cbam], 1)
        boundary = self.edf_net(layer1, high_feature)
        boundary = F.interpolate(boundary, size=x.size()[2:], mode='bilinear', align_corners=True)
        
        '''Final Mirror Map'''
        refine_rccl_map = torch.sigmoid(rccl_refine)
        refine_ssf_map = torch.sigmoid(ssf_refine)
        refine_sh_map = torch.sigmoid(sh_refine)
        boundary_map = torch.sigmoid(boundary)
        
        refine_feat = torch.cat([refine_rccl_map, refine_ssf_map, refine_sh_map, boundary_map], 1)
        final_predict = self.refine(refine_feat)
        
        if self.training:
            return boundary, final_predict
        # test mode
        else:
            rccl_layers = [rccl_4, rccl_3, rccl_2, rccl_1]
            ssf_layers = [ssf_4, ssf_3, ssf_2, ssf_1]
            sh_layers = [sh_4, sh_3, sh_2, sh_1]
            others = [boundary, final_predict]
            rccl_layers = [self.apply_sigmoid_if_needed(layer) for layer in rccl_layers]
            ssf_layers = [self.apply_sigmoid_if_needed(layer) for layer in ssf_layers]
            sh_layers = [self.apply_sigmoid_if_needed(layer) for layer in sh_layers]
            others = [self.apply_sigmoid_if_needed(layer) for layer in others]
    
        return (*rccl_layers, *ssf_layers, *sh_layers, *others, refine_rccl_map, refine_ssf_map, refine_sh_map)
    
    def output_zero(self, x):
        return x * torch.zeros_like(x)
    
    def apply_sigmoid_if_needed(self, tensor):
        return torch.sigmoid(tensor) if not self.training else tensor
    


