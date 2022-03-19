#!/usr/bin/env python
#-*- coding:utf-8 -*-

#############################################
# File Name: load.py
# Author: Haoyu Lu
# Mail: lhy1998@ruc.edu.cn
# Created Time:  2022-3-18 19:17:34
#############################################

import torch
import sys 

from ..models import build_network
from ..utils.config import cfg_from_yaml_file, cfg

def load_wenlan_model(load_checkpoint='../wenlan-video-model.pth', cfg_file='none', device='cpu'):

    cfg_from_yaml_file(cfg_file, cfg)

    cfg.MODEL.IMG_SIZE = 384
    cfg.MODEL.IS_EXTRACT = True
    cfg.DATASET.TEST_SET = 'test'

    model = build_network(cfg.MODEL)
    model.load_state_dict(torch.load(load_checkpoint))

    model = model.to(device)
    model.eval()

    return model