import os
import math
import sys

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as Func
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module

import torch.optim as optim




class ConvTemporalGraphical(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 bias=True):
        super(ConvTemporalGraphical,self).__init__()
        self.kernel_size = kernel_size
        

    def forward(self, x, A, A2):
        #A.size(0) == self.kernel_size == T == 8
        assert A.size(0) == self.kernel_size
        #x = self.conv(x)
        x1 = torch.einsum('nctv,tvw->nctw', (x, A))
        x2 = torch.einsum('nctv,tvw->nctw', (x, A2))
        lamda = 0.4
        beta = 0.6
        x = lamda * x1 + beta * x2
        return x.contiguous(), A
    

class st_gcn(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 use_mdn = False,
                 stride=1,
                 dropout=0,
                 residual=True):
        super(st_gcn,self).__init__()
        
        assert len(kernel_size) == 2
        assert kernel_size[0] % 2 == 1
        padding = ((kernel_size[0] - 1) // 2, 0)
        self.use_mdn = use_mdn

        self.gcn = ConvTemporalGraphical(in_channels, out_channels,kernel_size[1])

        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.PReLU(),
            nn.Conv2d(
                out_channels,
                out_channels,
                (kernel_size[0], 1),
                (stride, 1),
                padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )

        if not residual:
            self.residual = lambda x: 0

        elif (in_channels == out_channels) and (stride == 1):
            self.residual = lambda x: x

        else:
            self.residual = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=1,
                    stride=(stride, 1)),
                nn.BatchNorm2d(out_channels),
            )

        self.prelu = nn.PReLU()

    def forward(self, x, A, A2):
        res = self.residual(x)
        
        x, A = self.gcn(x, A, A2)
    
        x = self.tcn(x) + res
        
        if not self.use_mdn:
            x = self.prelu(x)
        return x, A

class electrical_stgcnn(nn.Module):
    def __init__(self,n_stgcnn =1,n_txpcnn=1,input_feat=3,output_feat=3,
                 obs_len=8,pred_seq_len=1,kernel_size=3):
        super(electrical_stgcnn,self).__init__()
        self.n_stgcnn= n_stgcnn
        self.n_txpcnn = n_txpcnn        
        self.st_gcns = nn.ModuleList()
        self.st_gcns.append(st_gcn(input_feat,output_feat,(kernel_size,obs_len)))
        for j in range(1,self.n_stgcnn):
            self.st_gcns.append(st_gcn(output_feat,output_feat,(kernel_size,obs_len)))
        self.tpcnns = nn.ModuleList()
        self.tpcnns.append(nn.Conv2d(obs_len,pred_seq_len,3,padding=1))
        for j in range(1,self.n_txpcnn):
            self.tpcnns.append(nn.Conv2d(pred_seq_len,pred_seq_len,3,padding=1))
            
        self.prelus = nn.ModuleList()
        for j in range(self.n_txpcnn):
            self.prelus.append(nn.PReLU())
        self.tpcnn_ouput = nn.Conv1d(15,1,1)


        
    def forward(self,v,a,a2):
        
        for k in range(self.n_stgcnn):
            v,a = self.st_gcns[k](v,a,a2)
        v = v.view(v.shape[0],v.shape[2],v.shape[1],v.shape[3])
        v = self.prelus[0](self.tpcnns[0](v))
        for k in range(1,self.n_txpcnn-1):
            v =  self.prelus[k](self.tpcnns[k](v)) + v
        v = v.view(v.shape[0],v.shape[1],v.shape[3],v.shape[2])
        v = v.view(v.shape[0],v.shape[1],v.shape[2]*v.shape[3])
        v = self.tpcnn_ouput(v.permute(0,2,1))
        return v,a
