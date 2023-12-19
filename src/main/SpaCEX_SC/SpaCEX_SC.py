import torch
import torch.nn as nn
from scipy.sparse import csr_matrix
import math
import random
from tqdm import tqdm
import SpaGCN as spg
import scanpy as sc
import pandas as pd
import numpy as np
import pickle
import json
import os,csv,re
import pandas as pd
import numpy as np
import math
import SpaGCN as spg
from scipy.sparse import issparse
import random, torch
import warnings
warnings.filterwarnings("ignore")
import matplotlib.colors as clr
import matplotlib.pyplot as plt

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

class SpaCEX_SC():
    def gene_select(cluster_labels, svg_score, selection_percentage=0.4):
        unique_labels = np.unique(cluster_labels)

        # new samples
        new_samples = []

        for label in unique_labels:
            current_samples = np.where(cluster_labels == label)[0]
            sorted_indices = np.argsort(svg_score[current_samples])[::-1]
            num_selected_genes = int(selection_percentage * len(current_samples))
            selected_genes = current_samples[sorted_indices[:num_selected_genes]]

            new_samples.extend(selected_genes)
            new_samples_indices = np.array(new_samples).reshape(-1)
        print("select {} genes".format(len(new_samples_indices)))

        return new_samples_indices
    
    def spatial_clustering(adata, new_samples_indices, n_clusters):
        adata = adata[~adata.obs['layer_guess_reordered_short'].isna()]
        adata = adata[~adata.obs['discard'].astype(bool), :]
        adata.obs['cluster'] = adata.obs['layer_guess_reordered_short'] 
        adata.var_names_make_unique()
        spg.prefilter_genes(adata, min_cells=3)  # avoiding all genes are zeros
        spg.prefilter_specialgenes(adata)
        adata.obs['x_array']=np.ceil((adata.obs['array_col']-adata.obs['array_col'].min())/2).astype(int)
        adata.obs['y_array']=adata.obs['array_row']-adata.obs['array_row'].min()
        #adata.X = adata.X.todense()
        x_array=adata.obs["x_array"].tolist()
        y_array=adata.obs["y_array"].tolist()
        x_pixel, y_pixel = adata.obsm['spatial'][:, 1], adata.obsm['spatial'][:, 0]
        adata.obs["x_pixel"] = x_pixel
        adata.obs["y_pixel"] = y_pixel
        sc.pp.normalize_per_cell(adata)
        sc.pp.log1p(adata)
        adata_new = adata[:, new_samples_indices]

        s=1
        b=49
        adj=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
        p=0.5 
        l=spg.search_l(p, adj, start=0.01, end=1000, tol=0.01, max_run=100)
        #For this toy data, we set the number of clusters=7 since this tissue has 7 layers
        r_seed=t_seed=n_seed=90
        res=spg.search_res(adata_new, adj, l, n_clusters, start=0.7, step=0.1, tol=5e-3, lr=0.05, max_epochs=20, r_seed=r_seed, t_seed=t_seed, n_seed=n_seed)
        
        clf=spg.SpaGCN()
        clf.set_l(l)
        #Set seed
        random.seed(r_seed)
        torch.manual_seed(t_seed)
        np.random.seed(n_seed)
        #Run
        clf.train(adata_new,adj,init_spa=True,init="louvain",res=res, tol=5e-3, lr=0.05, max_epochs=200)
        y_pred, prob=clf.predict()
        adata_new.obs["pred"]= y_pred
        adata_new.obs["pred"]=adata_new.obs["pred"].astype('category')
        #Do cluster refinement(optional)
        #shape="hexagon" for Visium data, "square" for ST data.
        adj_2d=spg.calculate_adj_matrix(x=x_array,y=y_array, histology=False)
        r_pred=spg.refine(sample_id=adata_new.obs.index.tolist(), pred=adata_new.obs["pred"].tolist(), dis=adj_2d, shape="hexagon")
        adata_new.obs["r_pred"]=r_pred
        adata_new.obs["r_pred"]=adata_new.obs["r_pred"].astype('category')
        
        return adata_new
    