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

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from numpy import linalg as LA
import networkx as nx
from tqdm import tqdm
import time
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler

def asim(p1,p2,attributes):
    distance = 0
    for i in range(attributes):
        distance = distance + (p1[i]-p2[i])**2
    NORM = math.sqrt(distance)
    if NORM ==0:
        return 0
    return 1/(NORM)

def anorm(p1,p2,attributes):
    distance = 0
    for i in range(attributes):
        distance = distance + (p1[i]-p2[i])**2
    NORM = math.sqrt(distance)
    return NORM

def aexp(p1,p2,attributes):
    distance = 0
    for i in range(attributes):
        distance = distance + (p1[i]-p2[i])**2
    NORM = math.exp(-math.sqrt(distance))
    return NORM

def seq_to_graph(seq_,norm_lap_matr = True):
    seq_len = seq_.shape[0] # time_sequence(T=1,2,4,8...)
    category = seq_.shape[1] # A/N=5
    attributes = seq_.shape[2] #D=3
    V = np.zeros((seq_len,category,attributes))
    A = np.zeros((seq_len,category,category)) 
    A2 = np.zeros((seq_len,category,category)) #kernel
    for s in range(seq_len):
        V[s] = seq_[s]
        A[s] = [[0,0.127,0.61,0.209,0.054],
                [0.092,0,0.533,0.243,0.131],
                [0.265,0.32,0,0.174,0.241],
                [0.19,0.305,0.365,0,0.14],
                [0.057,0.192,0.587,0.163,0]]
        for h in range(category):
            A2[s,h,h] = 1
            for k in range(h+1,category):
                l2_norm = aexp(seq_[s][h],seq_[s][k],attributes)
                A2[s,h,k] = l2_norm
                A2[s,k,h] = l2_norm
        if norm_lap_matr:
            G = nx.from_numpy_matrix(A[s,:,:])
            A[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()
            G = nx.from_numpy_matrix(A2[s,:,:])
            A2[s,:,:] = nx.normalized_laplacian_matrix(G).toarray()

    return torch.from_numpy(V).type(torch.float),\
           torch.from_numpy(A).type(torch.float),\
           torch.from_numpy(A2).type(torch.float)

def dataLoading(data_dir, groundTruth_col):
    dataset = read_csv(data_dir, usecols=range(0,groundTruth_col), engine='python').values.astype('float32')
    groundTruth = read_csv(data_dir, usecols=[groundTruth_col], engine='python').values.astype('float32')
    # normalize the dataset
    scaler = MinMaxScaler(feature_range=(0, 1))
    dataset = scaler.fit_transform(dataset)
    return dataset, groundTruth


class ElectricalDataset(Dataset):
    def __init__(
        self, data_dir, obs_len=8, pred_len=1, groundTruth_col=15, real_time = True, norm_lap_matr = True):
        super(ElectricalDataset, self).__init__()
        self.data_dir = data_dir
        self.obs_len = obs_len
        self.pred_len = pred_len 
        self.groundTruth_col = groundTruth_col
        if real_time: #real-time prediction
            self.seq_len = self.obs_len + self.pred_len - 1
        else: #anticipation
            self.seq_len = self.obs_len + self.pred_len
        self.norm_lap_matr = norm_lap_matr
        num_in_seq = []
        seq_list = []
        groundTruth_list = []
        dataset, groundTruth = dataLoading(data_dir, groundTruth_col)
        num_sequences = len(dataset) - self.seq_len + 1
        for idx in range(num_sequences):
            curr_seq_data = np.concatenate(dataset[idx:idx + self.seq_len], axis=0)
            curr_seq_data = np.reshape(curr_seq_data, (self.seq_len, 5, 3))
            num_in_seq.append(self.seq_len)
            seq_list.append(curr_seq_data)
            groundTruth_list.append(groundTruth[idx+self.seq_len-1])

        self.num_seq = len(seq_list)
        seq_list = np.concatenate(seq_list, axis=0)
        groundTruth_list = np.concatenate(groundTruth_list, axis=0)
        
        # Convert numpy -> Torch Tensor
        self.seq_list = torch.from_numpy(seq_list).type(torch.float)
        self.groundTruth_list = torch.from_numpy(groundTruth_list).type(torch.float)
        
        cum_start_idx = [0] + np.cumsum(num_in_seq).tolist()
        self.seq_start_end = [
            (start, end)
            for start, end in zip(cum_start_idx, cum_start_idx[1:])
        ]

        #Convert to Graphs
        self.v_obs = []
        self.A_obs = []
        self.A2_obs = []
        print("Processing Data .....")
        pbar = tqdm(total=len(self.seq_start_end)) 
        for ss in range(len(self.seq_start_end)):
            pbar.update(1)
            start, end = self.seq_start_end[ss]
            v_,a_,a2_ = seq_to_graph(self.seq_list[start:end,:],self.norm_lap_matr)
            self.v_obs.append(v_.clone())
            self.A_obs.append(a_.clone())
            self.A2_obs.append(a2_.clone())
        pbar.close()
        print("Data Processed .....")

    def __len__(self):
        return self.num_seq

    def __getitem__(self, index):

        out = [
            self.v_obs[index], self.A_obs[index], self.A2_obs[index],
            self.groundTruth_list[index]
        ]
        return out
