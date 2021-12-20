import os
import csv
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

from utils import * 
import pickle
import argparse
from torch import autograd
import torch.optim.lr_scheduler as lr_scheduler
from model import *

# Parameters
parser = argparse.ArgumentParser()
#Model specific parameters
parser.add_argument('--input_size', type=int, default=3)
parser.add_argument('--output_size', type=int, default=3)
parser.add_argument('--n_stgcnn', type=int, default=1, help='Number of ST-GCNN layers')
parser.add_argument('--n_txpcnn', type=int, default=1, help='Number of TXPCNN layers')
parser.add_argument('--kernel_size', type=int, default=3)

#Data specifc paremeters
parser.add_argument('--obs_seq_len', type=int, default=8)
parser.add_argument('--pred_seq_len', type=int, default=1)
parser.add_argument('--groundTruth_col', type=int, default=15)
parser.add_argument('--dataset', default='manual', help='PCA,Autoencoder,CNN')


#Training specifc parameters
parser.add_argument('--batch_size', type=int, default=64,
                    help='minibatch size')
parser.add_argument('--num_epochs', type=int, default=30,
                    help='number of epochs')  
parser.add_argument('--clip_grad', type=float, default=None,
                    help='gadient clipping')        
parser.add_argument('--lr', type=float, default=0.01,
                    help='learning rate')
parser.add_argument('--lr_sh_rate', type=int, default=150,
                    help='number of steps to drop the lr')  
parser.add_argument('--use_lrschd', action="store_true", default=False,
                    help='Use lr rate scheduler')
parser.add_argument('--tag', default='tag',
                    help='personal tag for the model ')
                    
args = parser.parse_args()



#Global Training Records 
metrics = {'train_loss':[],  'val_loss':[]}
constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}
evaluation_metrics = {'Accuracy':['Accuracy',], 'Precision':['Precision',], 'Recall':['Recall',], 'F1-Score':['F1-Score',]}

#Defining the model
model = electrical_stgcnn(n_stgcnn =args.n_stgcnn,n_txpcnn=args.n_txpcnn,
    obs_len=args.obs_seq_len,kernel_size=args.kernel_size,pred_seq_len=args.pred_seq_len).cuda()

print('*'*30)
print("Training initiating....")
print(args)


#Training settings 
optimizer = optim.SGD(model.parameters(),lr=args.lr)

if args.use_lrschd:
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_sh_rate, gamma=0.2)

checkpoint_dir = './checkpoint/'+args.tag+'/'

if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)
with open(checkpoint_dir+'args.pkl', 'wb') as fp:
    pickle.dump(args, fp)
    
data_set = './datasets/'+args.dataset+'/'


def train(iterator,epoch,loader_train):
    #global metrics, dset_train
    model.train()
    loss_batch = 0
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_train)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1

    for cnt,batch in enumerate(loader_train): 
        batch_count+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        V_obs,A_obs,A2_obs,groundTruth_list= batch
        optimizer.zero_grad()
        V_obs_tmp =V_obs.permute(0,3,1,2)
        V_pred,_ = model(V_obs_tmp,A_obs.squeeze(0),A2_obs.squeeze(0))
        V_pred = V_pred.squeeze()
        groundTruth_list = groundTruth_list.squeeze()

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = nn.MSELoss()(V_pred,groundTruth_list)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss = loss + l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True

            loss.backward()
            if args.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(),args.clip_grad)
            optimizer.step()
            #Metrics
            loss_batch += loss.item()
            #print('TRAIN:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)
            
    metrics['train_loss'].append(loss_batch/batch_count)
    
def vald(iterator,epoch,loader_val):
    p_p = 0
    n_n = 0
    p_n = 0
    n_p = 0
    threshold = 0.3
    model.eval()
    loss_batch = 0 
    batch_count = 0
    is_fst_loss = True
    loader_len = len(loader_val)
    turn_point =int(loader_len/args.batch_size)*args.batch_size+ loader_len%args.batch_size -1
    for cnt,batch in enumerate(loader_val): 
        batch_count+=1
        #Get data
        batch = [tensor.cuda() for tensor in batch]
        V_obs,A_obs,A2_obs,groundTruth_list= batch
        V_obs_tmp =V_obs.permute(0,3,1,2)
        V_pred,_ = model(V_obs_tmp,A_obs.squeeze(0),A2_obs.squeeze(0))
        V_pred = V_pred.squeeze()
        groundTruth_list = groundTruth_list.squeeze()
        
        if (V_pred>threshold and groundTruth_list>threshold):
            p_p = p_p + 1
        elif (V_pred>threshold and groundTruth_list<threshold):
            p_n = p_n + 1
        elif (V_pred<threshold and groundTruth_list<threshold):
            n_n = n_n + 1
        else:
            n_p = n_p + 1

        if batch_count%args.batch_size !=0 and cnt != turn_point :
            l = nn.MSELoss()(V_pred,groundTruth_list)
            if is_fst_loss :
                loss = l
                is_fst_loss = False
            else:
                loss = loss + l

        else:
            loss = loss/args.batch_size
            is_fst_loss = True
            #Metrics
            loss_batch += loss.item()
            #print('VALD:','\t Epoch:', epoch,'\t Loss:',loss_batch/batch_count)

    metrics['val_loss'].append(loss_batch/batch_count)
    if  metrics['val_loss'][-1]< constant_metrics['min_val_loss']:
        constant_metrics['min_val_loss'] =  metrics['val_loss'][-1]
        constant_metrics['min_val_epoch'] = epoch
        Accuracy = (p_p+n_n)/loader_len
        Precision = p_p / (p_p + p_n + 0.0000001)
        Recall = p_p / (p_p + n_p + 0.0000001)
        F1_Score = 2*Precision*Recall / (Precision + Recall + 0.0000001)
        evaluation_metrics['Accuracy'].append(Accuracy)
        evaluation_metrics['Precision'].append(Precision)
        evaluation_metrics['Recall'].append(Recall)
        evaluation_metrics['F1-Score'].append(F1_Score)
        #print(evaluation_metrics)
        torch.save(model.state_dict(),checkpoint_dir+str(iterator)+'/'+str(epoch)+'.pth')  # OK


def main():
    #os.environ["CUDA_VISIBLE_DEVICES"]="1"
    print('Training started ...')
    for iterator in range(10):
        global metrics, constant_metrics, evaluation_metrics
        iterator = iterator + 1
        if iterator != 1:
            metrics = {'train_loss':[],  'val_loss':[]}
            constant_metrics = {'min_val_epoch':-1, 'min_val_loss':9999999999999999}
            evaluation_metrics = {'Accuracy':['Accuracy',], 'Precision':['Precision',], 'Recall':['Recall',], 'F1-Score':['F1-Score',]}
        dset_train = ElectricalDataset(
            data_set+'Cy'+str(iterator)+'/trainSet.csv',
            obs_len=args.obs_seq_len,
            pred_len=args.pred_seq_len,
            groundTruth_col = args.groundTruth_col,
            #real_time=True,
            norm_lap_matr=True)

        loader_train = DataLoader(
            dset_train,
            batch_size=1, #This is irrelative to the args batch size parameter
            shuffle = False,
            num_workers=0)

        dset_val = ElectricalDataset(
            data_set+'Cy'+str(iterator)+'/testSet.csv',
            obs_len=args.obs_seq_len,
            pred_len=args.pred_seq_len,
            groundTruth_col = args.groundTruth_col,
            #real_time=True,
            norm_lap_matr=True)

        loader_val = DataLoader(
            dset_val,
            batch_size=1, #This is irrelative to the args batch size parameter
            shuffle =False,
            num_workers=0)

        for epoch in range(args.num_epochs):
            train(iterator,epoch,loader_train)
            vald(iterator,epoch,loader_val)
            if args.use_lrschd:
                scheduler.step()
 
        with open(checkpoint_dir+str(iterator)+"/evaluation_metrics.csv",'w',newline='') as t:
                writer=csv.writer(t)
                writer.writerow(evaluation_metrics['Accuracy'])
                writer.writerow(evaluation_metrics['Precision'])
                writer.writerow(evaluation_metrics['Recall'])
                writer.writerow(evaluation_metrics['F1-Score'])  

if __name__ == '__main__':
    main()



