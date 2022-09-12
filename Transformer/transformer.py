import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
class LayerNorm(nn.Module):
    def __init__(self,feature,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2 = nn.Parameter(torch.ones(feature))
        self.b_2 = nn.Parameter(torch.zeros(feature))
        self.eps = eps
    def forward(self,x):
        mean = x.mean(-1,keepdim=True)
        std = x.std(-1,keepdim=True)
        return self.a_2 * (x-mean)/(std+self.eps)+self.b_2
class SublayerConnection(nn.Module):
    def __init__(self,size,dropout=0.1):
        super(SublayerConnection, self).__init__()
        self.layer_norm = LayerNorm(size)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self,x,sublayer):
        '''
        :param x:上层的输入
        :param sublayer:上层
        :return:
        '''
        return self.dropout(self.layer_norm(x + sublayer(x)))