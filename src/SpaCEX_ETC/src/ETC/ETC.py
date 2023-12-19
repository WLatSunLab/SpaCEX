import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import torch.nn as nn
import torch
import math
import scanpy as sc
import json
import os
import torch.nn.functional as F
from scipy.sparse import csr_matrix
import torchvision.transforms as transforms
import warnings
import pickle
import torch
warnings.filterwarnings("ignore")

class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss
    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """
    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum

def pearson_correlation(batch1, batch2):
    batch1_flat = batch1.view(batch1.shape[0], -1)
    batch2_flat = batch2.view(batch2.shape[0], -1)

    mean_batch1 = torch.mean(batch1_flat, dim=1, keepdim=True)
    mean_batch2 = torch.mean(batch2_flat, dim=1, keepdim=True)

    diff_batch1 = batch1_flat - mean_batch1
    diff_batch2 = batch2_flat - mean_batch2

    numerator = torch.sum(diff_batch1 * diff_batch2, dim=1)

    denominator = torch.sqrt(torch.sum(diff_batch1**2, dim=1) * torch.sum(diff_batch2**2, dim=1))

    pcc = numerator / denominator

    return pcc

import random
def hard_shrink_relu(x, lambd=0, epsilon=1e-12):
    '''
    relu based hard shrinkage function, only works for positive values
    '''
    x = (F.relu(x-lambd) * x) / (torch.abs(x - lambd) + epsilon)
    return x

class MemoryUnit(nn.Module):
    def __init__(self, mem_dim, z_dim, shrink_thres=0.005, tem=0.5):
        super().__init__()
        self.mem_dim = mem_dim
        self.z_dim = z_dim
        self.shrink_thres = shrink_thres
        self.tem = tem
        self.register_buffer("mem", torch.randn(self.mem_dim, self.z_dim))
        self.register_buffer("mem_ptr", torch.zeros(1, dtype=torch.long))
        self.update_size = 64
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.mem.size(1))
        self.mem.data.uniform_(-stdv, stdv)

    @torch.no_grad()
    def update_mem(self, z):
        n_obs = z.shape[0]
        idx = random.sample([i for i in range(n_obs)], self.update_size)
        z =  z[idx, :]
    
        batch_size = z.shape[0]  # z, B x C
        ptr = int(self.mem_ptr)
        assert self.mem_dim % batch_size == 0

        # replace the keys at ptr (dequeue and enqueue)
        self.mem[ptr:ptr + batch_size, :] = z  # mem, M x C
        ptr = (ptr + batch_size) % self.mem_dim  # move pointer

        self.mem_ptr[0] = ptr

    def attention(self, input):
        att_weight = torch.mm(input, self.mem.T)  # input x mem^T, (BxC) x (CxM) = B x M
        att_weight = F.softmax(att_weight/self.tem, dim=1)  # B x M

        # ReLU based shrinkage, hard shrinkage for positive value
        if (self.shrink_thres > 0):
            att_weight = hard_shrink_relu(att_weight, lambd=self.shrink_thres)
            att_weight = F.normalize(att_weight, p=1, dim=1)

        output = torch.mm(att_weight, self.mem)  # AttWeight x mem, (BxM) x (MxC) = B x C
        return output

    def forward(self, x):
        x = self.attention(x)
        
        return x

class Generator(nn.Module): # Generate(281, 204)
    def __init__(self, out_dim=[256, 512, 1024, 281*204]):
        super(Generator, self).__init__()
        self.Memory = MemoryUnit(out_dim[0], out_dim[1], shrink_thres=0.005, tem=0.5)
        self.encoder = nn.Sequential(
            nn.Linear(64, out_dim[0]),
            nn.LeakyReLU(),
            nn.LayerNorm(out_dim[0]),
            nn.Linear(out_dim[0], out_dim[1])
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(out_dim[1], out_dim[2]),
            nn.LeakyReLU(),
            nn.LayerNorm(out_dim[2]),
            nn.Linear(out_dim[2], out_dim[3]),
        )
        
        self.idf = nn.Sequential(
            nn.Linear(64, out_dim[1]),
        )

    def forward(self, x):
        z = self.encoder(x)
        mem_img = self.Memory(z)
        idf = self.idf(x)
        img = mem_img + idf
        img = self.decoder(img)
        img = img.view(-1, 1, 281, 204)
        
        return img, z

class Discriminator(nn.Module):  # input(-1, 1, 17, 20)
    def __init__(self):
        super(Discriminator,self).__init__()
        self.main = nn.Sequential(
            nn.Linear(281*204,1024),
            nn.LeakyReLU(), # x小于零是是一个很小的值不是0，x大于0是还是x
            nn.LayerNorm(1024),
            nn.Linear(1024,256),

        )
        self.idf = nn.Sequential(
            nn.Linear(281*204, 256),
            nn.LeakyReLU(),
            nn.LayerNorm(256)
        )
        self.sig = nn.Sequential(
            nn.Linear(256, 128),
            nn.LeakyReLU(),
            nn.LayerNorm(128),
            nn.Linear(128, 64),
            #nn.LeakyReLU(),
            #nn.LayerNorm(64),
            #nn.Linear(64, 1)
            
        )
    def forward(self, x):
        x = x.view(-1,281*204)
        img = self.main(x)
        idf = self.idf(x)
        img = img + idf
        img = self.sig(img)
        #img = torch.sum(img, dim=1)  
        
        return img