import numpy as np
import rpy2.robjects as robjects
from rpy2.robjects import packages
from tqdm import tqdm
import torch

def simu_zinb(adata,maxiter=9,threshold=0.001,alpha=0.99):
    X =np.exp(adata.X)-1
    n =X.shape[0]
    
    x_bar=np.mean(X,axis=0)
    s2_bar=np.var(X,axis=0)
    p_bar=np.sum(X==0,axis=0)/n
    o=p_bar*alpha
    for i in range(maxiter):#循环9次比较好
        m=np.divide(x_bar,(1-o))
        V=np.divide(s2_bar-np.multiply(np.multiply(o,(1-o)),np.square(m)),1-o) 
        V[V<1e-5]=1e-5
        s=np.divide(np.square(m),(V-m)) 
        s[s<1]=1
        x=np.divide(s,m+s)
        x[x<1e-6]=1e-6
        o1=np.divide(p_bar-np.power(x,s),1-np.power(x,s))
        #o = np.squeeze(o)
        #o1 = np.squeeze(o1)
        o1[np.where(np.square(o-o1)<threshold)]=o[np.where(np.square(o-o1)<threshold)]
        o=o1
    return m,s,o

def get_svg_score(z, dataset, adata, model, sim_times=4):
    m,s,o=simu_zinb(adata)
    m = np.array(m).reshape(-1)
    s = np.array(s).reshape(-1)
    o = np.array(o).reshape(-1)
    mask=np.zeros(shape=(adata.obs['array_y'].max()+1,adata.obs['array_x'].max()+1)) 
    
    for j,row_col in enumerate(zip(adata.obs['array_y'],adata.obs['array_x'])):
        row_ix,col_ix=row_col
        mask[row_ix,col_ix]=1
    
    zim=packages.importr('ZIM')
    zinb=robjects.r['rzinb']
    dis=torch.zeros((adata.shape[1]))
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    for i in tqdm(range(sim_times)):
        simu = [np.array(zinb(dataset.shape[1]*dataset.shape[2],float(x),float(y),float(z))).reshape(dataset.shape[1],dataset.shape[2]) for x,y,z in zip(s,m,o)]
        simu=torch.tensor(np.array(simu)*mask,dtype=torch.float32).unsqueeze(1).to(device)
        simu[simu<0]=0
        simu=torch.log(simu+1)
        data = simu.reshape(-1, 1, dataset.shape[1], dataset.shape[2])
        batch_size = 256
        z1 = []
        with torch.no_grad():
            for i in range(0, len(data), batch_size):
                data_batch = torch.tensor(data[i:i+batch_size], dtype=torch.float32).to(device)
                _, z1_part, _, _, _ = model(data_batch)
                z1.append(z1_part)
        z1 = torch.cat(z1, dim=0)
        z1 = z1.detach().cpu()
        z = z.detach().cpu()
        dis+=torch.square(z-z1).sum(1)

    score = np.array((dis/4))
    
    return score