import torch
import torch.nn as nn
from SpaCEX.src.SpaCEX_ETC.src.main import driver
from SpaCEX.src.SpaCEX_ETC.src.main.clustering.DEC import DEC
from SpaCEX.src.SpaCEX_ETC.src.main._config import Config
from scipy.sparse import csr_matrix
from SpaCEX.src.SpaCEX_ETC.src.ETC.ETC import AutomaticWeightedLoss, Generator, Discriminator
import torch.nn.functional as F
import math
import random
from tqdm import tqdm
import SpaGCN as spg
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import json
import os

'''
What you should input
dataset: gene image data with size [N, 1, 72, 59]
total: gene similarity matrix with size [N, N]

What you will get return
model: encoder that haved been trained
y_pred: label that SpaCEX generative
embedding: embedding that generatived by encoder

Others
If you wanna get other return such as x_bar or parameters of SMM, just rewrite DEC to get what you want.
'''

class SpaCEX_ETC():
    def get_data(data='10x', data_type='image'):
        if data == '10x':
            path_file = 'SpaCEX/data/mEmb/10x_mEmb_matrix.dat'
            with open(path_file,'rb') as f:
                all_gene_exp_matrices = pickle.load(f)
            all_gmat_v = {k:all_gene_exp_matrices[k].todense()[25:60,40: ]for k in list(all_gene_exp_matrices.keys())}
            key_v = list(all_gmat_v.keys())
            dataset_v=np.array(list(all_gmat_v.values()))
            
            return key_v, dataset_v
        
        if data =='sqf' and data_type == 'adata':
            path_file = 'SpaCEX/data/mEmb/sqf_mEmb_adata.h5ad'
            adata = sc.read_h5ad(path_file)
            
            return adata
        
        if data == 'sqf' and data_type == 'image':
            path_file = 'SpaCEX/data/mEmb/sqf_mEmb_matrix.dat'
            with open(path_file,'rb') as f:
                all_gene_exp_matrices = pickle.load(f)
            all_gmat_m = {k:all_gene_exp_matrices[k].todense()[25:60,40: ]for k in list(all_gene_exp_matrices.keys())}
            key_m = list(all_gmat_m.keys())
            dataset_m=np.array(list(all_gmat_m.values()))
            
            return key_m, dataset_m
    
    def data_process(adata):
        adata.X=csr_matrix(adata.X)

        coor_raw = adata.obsm['spatial']
        coor = np.zeros(coor_raw.shape)
        max_normalized_x = coor_raw[:, 0].max()
        max_normalized_y = coor_raw[:, 1].max()
        min_normalized_x = coor_raw[:, 0].min()
        min_normalized_y = coor_raw[:, 1].min()
        coor[:, 0] = coor_raw[:, 0] - min_normalized_x
        coor[:, 1] = coor_raw[:, 1] - min_normalized_y
        coor = np.ceil(coor*40)

        all_genes = adata.var.index.values
        adata.obs['array_x']=np.ceil(coor[:, 0]).astype(int)
        adata.obs['array_y']=np.ceil(coor[:, 1]).astype(int)
        all_gene_exp_matrices = {}
        shape = (adata.obs['array_y'].max()+1, adata.obs['array_x'].max()+1)
        i = 0
        for gene in all_genes:
            g_matrix = np.zeros(shape=shape)

            g = adata[:,gene].X.todense().tolist()
            #c = adata.obsm['spatial']
            for i,row_col in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
                row_ix,col_ix = row_col
                g_matrix[row_ix,col_ix] += g[i][0]
            all_gene_exp_matrices[gene] = csr_matrix(g_matrix)
        all_gmat_m = {k:all_gene_exp_matrices[k].todense() for k in list(all_gene_exp_matrices.keys())}
        key_m = list(all_gmat_m.keys())
        dataset_m=np.array(list(all_gmat_m.values()))
        
        return adata, key_m, dataset_m
    
    def load_model():
        config = Config(dataset='Mouse image', model='MAE').get_parameters()
        cuda = torch.cuda.is_available()
        device = torch.device("cuda" if cuda else "cpu")
        model = driver.model('Mouse image', 'MAE', config)
        model.to(device)
        model.load_state_dict(torch.load('SpaCEX/model_pretrained/SpaCEX_ETC.pkl'))
        
        return model
    
    def data_filter(key_v, dataset_v, key_m, dataset_m):
        all_gmat_v2m = {}
        for i in range(len(dataset_v)):
            if key_v[i] in key_m:
                all_gmat_v2m.update({key_v[i]:dataset_v[i]})
        key_v = list(all_gmat_v2m.keys())
        dataset_v = np.array(list(all_gmat_v2m.values()))
        
        all_gmat_m2v = {}
        for i in range(len(dataset_m)):
            if key_m[i] in key_v:
                all_gmat_m2v.update({key_m[i]:dataset_m[i]})
        dataset_m = np.array(list(all_gmat_m2v.values()))
        key_m = list(all_gmat_m2v.keys())
        
        return all_gmat_v2m, all_gmat_m2v

    
    def train_ETC(adata, all_gmat_v2m, all_gmat_m2v, model):
        model1 = model
        key_v = list(all_gmat_v2m.keys())
        dataset_v = np.array(list(all_gmat_v2m.values()))
        key_m = list(all_gmat_m2v.keys())
        dataset_m = np.array(list(all_gmat_m2v.values()))
        
        device='cuda' if torch.cuda.is_available() else 'cpu'
        gen = Generator().to(device)
        dis = Discriminator().to(device)

        awl = AutomaticWeightedLoss(2)
        awl2 = AutomaticWeightedLoss(3)
        awl3 = AutomaticWeightedLoss(3)
        
        d_optim = torch.optim.RMSprop(dis.parameters(),lr=1e-3)
        g_optim = torch.optim.RMSprop([
            {'params': gen.parameters()},
            {'params': awl.parameters(), 'weight_decay': 0}
        ], lr=1e-4)
        model_optim2 = torch.optim.Adam([
            {'params': model1.parameters()},
            {'params': awl2.parameters(), 'weight_decay': 0}
        ], lr=1e-3)
        model_optim3 = torch.optim.Adam([
            {'params': model1.parameters()},
            {'params': awl3.parameters(), 'weight_decay': 0}
        ], lr=1e-4)
        
        # Rearrange viuium according to seqfish
        sorted_indices = sorted(range(len(key_v)), key=lambda i: key_v[i])
        key_v = [key_v[i] for i in sorted_indices]
        dataset_v = dataset_v[sorted_indices]
        
        x_y = np.array([adata.obs['array_y'].values, adata.obs['array_x'].values]).T
        res = np.zeros(dataset_m.shape[1:])
        for i in range(len(x_y)):
            res[x_y[i, 0], x_y[i, 1]] = 1
        res = torch.tensor(res, dtype=float).unsqueeze(0).to(device)
        
        clip_value = 1e-2
        batch_size = 64
        #lap_m = lap[:300]
        dataset_m_n = torch.tensor(dataset_m[:300])
        dataset_m_n = dataset_m_n.to(torch.float32)
        dataset_m_n = dataset_m_n.unsqueeze(1)
    
        dataset_v_n = torch.tensor(dataset_v[:300])
        dataset_v_n = dataset_v_n.to(torch.float32)
        dataset_v_n = dataset_v_n.unsqueeze(1)
        
        batch_num = len(dataset_v_n)//batch_size+1
        model1.train()
        for epoch in tqdm(range(100)):

            for i in range(0, len(dataset_m_n), batch_size):
                img = dataset_m_n[i:i+batch_size]
                img = img.to(device)
                res_part = res.expand(img.shape[0], -1, -1, -1).to(torch.float32)

        
                _, emb_part, d_recon_loss, _, _ = model1(dataset_v_n[i:i+batch_size].to(device))

                fake_img, _ = gen(emb_part)
                fake_img = fake_img
                fake_output = dis(fake_img)
                real_output = dis(img)
        
                _, emb_part, recon_loss, _, _ = model1(dataset_v_n[i:i+batch_size].to(device))
        
                fake_img, _ = gen(emb_part)
                fake_img = fake_img.detach()

                fake_output = dis(fake_img)
                real_output = dis(img)
                d_loss = -torch.mean(real_output) + torch.mean(fake_output)
                d_optim.zero_grad()
                d_loss.backward()
                d_optim.step()
        

                for p in dis.parameters():
                    p.data.clamp_(-clip_value, clip_value)
        
                _, emb_part, g_recon_loss, _, _ = model1(dataset_v_n[i:i+batch_size].to(device))
        
                fake_img, _ = gen(emb_part)
                fake_output = dis(fake_img)
        
                g_ce_loss = -torch.mean(fake_output)
                g_mse_loss = torch.mean(torch.abs(fake_img - img))

                model_optim3.zero_grad()
                model_loss3 = awl3(g_mse_loss, g_recon_loss, g_ce_loss)
                model_loss3.backward()
                model_optim3.step()
        
                _, emb_part, recon_loss, _, _ = model1(dataset_v_n[i:i+batch_size].to(device))
        
                emb_part = emb_part.detach()
                fake_img, z = gen(emb_part)
                fake_output = dis(fake_img)
        
                g_ce_loss = -torch.mean(fake_output)
                g_mse_loss = torch.mean(torch.abs(fake_img - img))
                g_loss = awl(g_ce_loss, g_mse_loss)
        
                g_optim.zero_grad() 
                g_loss.backward()
                g_optim.step()
        
                if i != 0:
                    gen.Memory.update_mem(z_old)
                else:
                    gen.Memory.update_mem(z)
                z_old = z
        
        emb_part = []
        total_samples = len(dataset_v)
        batch_size = 30
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_data = dataset_v[i:i+batch_size]  # Get a batch of data
                batch_data = torch.Tensor(batch_data).unsqueeze(1).to(device)
                _, batch_result_emb, _, _, _ = model1(batch_data)
                emb_part.append(batch_result_emb)
        SGEs = torch.cat(emb_part, dim=0)

        return gen, SGEs
    
    def sqf_gen(Gen, SGEs, adata):
        img_gen, _ = Gen(SGEs)
        img_gen = img_gen.to('cpu')
        img_gen1 = img_gen.detach().numpy()
        x_y = np.array([adata.obs['array_y'].values, adata.obs['array_x'].values]).T
        img_gen1 = img_gen1.reshape(-1, 281, 204)
        img_gen_norm = np.zeros(img_gen1.shape)
        for i in range(len(img_gen1)):
            img_gen_norm[i][x_y[:, 0], x_y[:, 1]] = img_gen1[i][x_y[:, 0], x_y[:, 1]]
        img_gen_seqfish = img_gen_norm[301:]
        img_gen_seqfish = img_gen_seqfish.reshape(-1, 281, 204)
        img_gen_seqfish[np.where(img_gen_seqfish<0)] = 0
        #img_gen_norm[np.where(img_gen_norm<0)] = 0
        
        return img_gen_norm
    
    def sequential_gen():
        print(1)
        
        
    def train(dataset, dataset_denoise=None, total=None, pretrain=False):
        config = Config(dataset='Mouse image', model='MAE').get_parameters()
        cuda = torch.cuda.is_available()
        print("use cuda: {}".format(cuda))
        device = torch.device("cuda" if cuda else "cpu")
        model = driver.model('Mouse image', 'MAE', config)
        model.to(device)
        if config['decoder'] == 'Gene image':
            model.pretrain(dataset = dataset, 
                           dataset_denoise = dataset_denoise, 
                           batch_size=config['batch_size'], 
                           lr=config['lr'])
        else:
            model.pretrain(dataset = dataset,
                           dataset_denoise = dataset_denoise, 
                           batch_size=256, 
                           lr=1e-3,
                           pretrain=False)
        y_pred, z, model= DEC(model, dataset, total = total, config = config)

        return y_pred, model

