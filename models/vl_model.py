#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: vl_model.py
# Author: Haoyu Lu
# Mail: lhy1998@ruc.edu.cn
# Created Time:  2022-3-18 19:17:34
#############################################

import torch
import torch.nn as nn
import torchvision
from transformers import AutoTokenizer
from .fakeTransformer import FakeTransformer
from .bert import Bert


import torch.nn.functional as F
import timm

import numpy as np
import math




class ImgLearnableEncoder(nn.Module):
    def __init__(self, model_cfg, args):
        super(ImgLearnableEncoder, self).__init__()

        
        self.backbone = timm.create_model(model_cfg.CNN, pretrained=False)

        self.model_cfg = model_cfg
        self.learnable = nn.ModuleDict()

        img_encoder_layer = nn.TransformerEncoderLayer(d_model=self.model_cfg.IMG_FEATURE_DIM, nhead=self.model_cfg.IMG_TRANSFORMER_HEAD)
        self.learnable['imgAtt'] = nn.TransformerEncoder(img_encoder_layer, num_layers=self.model_cfg.IMG_TRANSFORMER_LAYER)
        self.learnable['imgFC'] = FakeTransformer(model_cfg.IMG_FEATURE_DIM, model_cfg.HIDDEN_DIM_1, model_cfg.HIDDEN_DIM_2)
        self.learnable['max_pool'] = nn.Sequential(nn.AvgPool2d(model_cfg.ROI_GRID_SIZE, stride=1)) 

        # self.learnable['final_mlp'] = FakeTransformer(model_cfg.HIDDEN_DIM_2*2, model_cfg.HIDDEN_DIM_2, model_cfg.HIDDEN_DIM_2)


    def roi_grid_pool(self, spatial_features_2d, rois):
        """
        Args:
            rois: (B, num_rois, 4)
            spatial_features_2d: (B, C, H, W)
        Returns:
            pooled_features : (B, num_rois, C)
        """
        batch_size = spatial_features_2d.size(0)
        rois = rois.detach()
        height, width = spatial_features_2d.size(2), spatial_features_2d.size(3)
        down_sample_ratio = self.model_cfg.IMG_SIZE / height
        pooled_features_list = []
        torch.backends.cudnn.enabled = False
        for b_id in range(batch_size):
            # Map global boxes coordinates to feature map coordinates
            x1 = rois[b_id, :, 0] / down_sample_ratio
            y1 = rois[b_id, :, 1] / down_sample_ratio
            x2 = rois[b_id, :, 2] / down_sample_ratio
            y2 = rois[b_id, :, 3] / down_sample_ratio
            angle = torch.zeros((1), device=spatial_features_2d.device)
            cosa = torch.cos(angle)
            sina = torch.sin(angle)

            theta = torch.stack((
                (x2 - x1) / (width - 1) * cosa, (x2 - x1) / (width - 1) * (-sina), (x1 + x2 - width + 1) / (width - 1),
                (y2 - y1) / (height - 1) * sina, (y2 - y1) / (height - 1) * cosa, (y1 + y2 - height + 1) / (height - 1)
            ), dim=1).view(-1, 2, 3).float()

            grid_size = self.model_cfg.ROI_GRID_SIZE
            grid = nn.functional.affine_grid(
                theta,
                torch.Size((rois.size(1), spatial_features_2d.size(1), grid_size, grid_size))
            )

            pooled_features = nn.functional.grid_sample(
                spatial_features_2d[b_id].unsqueeze(0).expand(rois.size(1), spatial_features_2d.size(1), height, width),
                grid
            )
            pooled_features = self.learnable['max_pool'](pooled_features)
            pooled_features_list.append(pooled_features.squeeze())

        torch.backends.cudnn.enabled = True
        pooled_features = torch.stack(pooled_features_list, dim=0)

        return pooled_features

    def forward(self, imgFea, image_boxs):
        imgFea = self.backbone.forward_features(imgFea)
        imgFea = self.roi_grid_pool(imgFea, image_boxs)
        imgFea = F.normalize(imgFea, p=2, dim=-1)

        imgFea = self.learnable['imgAtt'](imgFea.transpose(0, 1)).transpose(0,1)    # TODO
        imgFea = imgFea.mean(1)
        imgFea = self.learnable['imgFC'](imgFea)


        return imgFea


class TextLearnableEncoder(nn.Module):
    def __init__(self, model_cfg):
        super(TextLearnableEncoder, self).__init__()

        self.backbone = Bert(model_cfg)
        self.model_cfg = model_cfg      # TODO: model_cfg

        self.learnable = nn.ModuleDict()
        
        text_encoder_layer = nn.TransformerEncoderLayer(d_model=model_cfg.TEXT_FEATURE_DIM, nhead=model_cfg.TEXT_TRANSFORMER_HEAD)
        self.learnable['textAtt'] = nn.TransformerEncoder(text_encoder_layer, num_layers=model_cfg.TEXT_TRANSFORMER_LAYER)

        self.learnable['textFC'] = FakeTransformer(model_cfg.TEXT_FEATURE_DIM, model_cfg.HIDDEN_DIM_1, model_cfg.HIDDEN_DIM_2)


    def forward(self, textFea, maskTexts):
        textFea = self.backbone(textFea)
        textFea = F.normalize(textFea, p=2, dim=-1)
        if self.model_cfg.TEXT_TRANSFORMER_LAYER != 0:
            textFea = self.learnable['textAtt'](textFea.transpose(0, 1), src_key_padding_mask=(maskTexts == 0)).transpose(0,1) 
        tmpMask = torch.where(maskTexts == 1, torch.tensor([1.0], device=maskTexts.device),
                              torch.tensor([0.0], device=maskTexts.device))
        textFea = (textFea * tmpMask.unsqueeze(-1)).sum(dim=1) / tmpMask.sum(dim=1).unsqueeze(-1)  # (bs, dim)
        textFea = self.learnable['textFC'](textFea)

        return textFea


class VL_model(nn.Module):

    def __init__(self, model_cfg, args):
        super(VL_model, self).__init__()

        self.model_cfg = model_cfg
        self.num_frame = model_cfg.num_frame

        self.learnable = nn.ModuleDict()
        self.learnable['imgencoder'] = ImgLearnableEncoder(model_cfg, args)
        self.learnable['textencoder'] = TextLearnableEncoder(model_cfg)
        
        video_layer = nn.TransformerEncoderLayer(d_model=model_cfg.HIDDEN_DIM_2, nhead=self.model_cfg.IMG_TRANSFORMER_HEAD)
        self.learnable['videoAtt'] = nn.TransformerEncoder(video_layer, num_layers=self.model_cfg.IMG_TRANSFORMER_LAYER)

        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')

    def imgfea2videofea(self,t):
        t = t.reshape(-1, self.num_frame, t.shape[1])
        # t = self.learnable['videoAtt'](t.transpose(0,1)).transpose(0,1)
        v = t.mean(dim = 1)
        return v       


    def forward(self, imgFea, image_boxs, texts, maskTexts):
        with torch.no_grad():
            imgFea = self.learnable['imgencoder'](imgFea, image_boxs) # <bsz, img_dim>
            imgFea = self.imgfea2videofea(imgFea)
            textFea = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
            
            imgFea = F.normalize(imgFea, p=2, dim=-1)
            textFea = F.normalize(textFea, p=2, dim=-1)

            return [imgFea, textFea]


    def encode_video(self, imgFea, image_boxs):
        with torch.no_grad():
            imgFea = self.learnable['imgencoder'](imgFea, image_boxs) # <bsz, img_dim>
            imgFea = self.imgfea2videofea(imgFea)
            imgFea = F.normalize(imgFea, p=2, dim=-1)
            return imgFea

    def encode_text(self, texts, maskTexts):
        with torch.no_grad():
            textFea = self.learnable['textencoder'](texts, maskTexts) # <bsz, img_dim>
            textFea = F.normalize(textFea, p=2, dim=-1)
            return textFea



        

