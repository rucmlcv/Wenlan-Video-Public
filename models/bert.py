import torch
import torch.nn as nn
from transformers import AutoModel


class Bert(nn.Module):

    def __init__(self, cfg):
        super(Bert, self).__init__()
        self.cfg = cfg
        self.bert =  AutoModel.from_pretrained(cfg.ENCODER) 

    def forward(self, x): 
        y = self.bert(x, return_dict=True).last_hidden_state
        return y
