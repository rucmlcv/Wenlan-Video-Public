import os
import numpy as np
import torch
import torch.utils.data as data
import torchvision.transforms as transforms

import json
import random


import jsonlines
import argparse

from PIL import ImageFilter
from transformers import AutoTokenizer
from PIL import Image
from PIL import ImageFile


def visual_transforms_box(is_train=True, new_size=384):
    mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((new_size, new_size)),
        normalize])




def getLanMask(seq_lens, max_len):
    # seq_lens (bs)
    mask = torch.ones((seq_lens.size(0), max_len))  # (bs, max_len)
    idxs = torch.arange(max_len).unsqueeze(dim=0)  # (1, max_len)
    seq_lens = seq_lens.unsqueeze(-1)  # (bs, 1)
    mask = torch.where(idxs < seq_lens, mask, torch.Tensor([0.0]))
    return mask



class wenlan_transforms():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
    
    def text_transform(self, text, max_length=50, device='cpu'):
        # text_transform = tokenizer #AutoTokenizer.from_pretrained('hfl/chinese-roberta-wwm-ext-large')
        text_info = self.tokenizer(text, padding='max_length', truncation=True,
                                        max_length=max_length, return_tensors='pt')
        text = text_info.input_ids.reshape(-1).to(device)
        text_len = torch.sum(text_info.attention_mask)
        textMask = getLanMask(text_len.unsqueeze(0), max_length).squeeze(0).to(device)

        return text, textMask
    
    def video_transform(self, video_path, device='cpu'):
        visual_transform = visual_transforms_box(False, 384)

        image_boxes = []
        images = []
        sorted_frames = os.listdir(video_path)
        for frame in sorted_frames:

            img_path = os.path.join(video_path, frame)
            image = Image.open(img_path).convert('RGB')
            image = visual_transform(image)
            images.append(image)
            img_box_s = []


            for grid_num in [1, 5]:
                for i in range(grid_num):
                    for j in range(grid_num):
                        img_box_s.append(torch.from_numpy(np.array(
                            [i * (384 / grid_num), j * (384 / grid_num), (i + 1) * (384 / grid_num),
                                (j + 1) * (384 / grid_num)])))
            image_boxes.append(torch.stack(img_box_s, 0))

        image_boxes = torch.stack(image_boxes, 0).to(device)
        images = torch.stack(images, 0).to(device)


        return images, image_boxes
