import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from SIGEL.src.main.SMM import initialize_SMM_parameters,calculate_xi
from SIGEL.src.main.EM import EM_algorithm
from SIGEL.src.main.LH import likelihood, regularization, size
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from SIGEL.src.main.LH import likelihood, regularization, size
from torch.nn import Linear
from tqdm import tqdm

from sklearn.cluster import MiniBatchKMeans




def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def patchify(imgs):
    """
    imgs: (N, 1, H, W)
    x: (N, L, patch_size**2 *1)
    """
    p = 4  # 4
    h = imgs.shape[2] // p  # 18
    w = imgs.shape[3] // p  # 14
    imgs = imgs[:, :, :h * p, :w * p]  # [:, :, :18, :14]
    x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
    x = torch.einsum('nchpwq->nhwpqc', x)
    x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))  # 在这里emb_dim=16
    return x

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
    
def caculate_rloss(imgs, pred, mask):
    """
    imgs: [N, 1, H, W]
    pred: [N, L, p*p*1]
    mask: [N, L], 0 is keep, 1 is remove,
    """
    target = patchify(imgs)
    norm_pix_loss = False
    if norm_pix_loss:
        mean = target.mean(dim=-1, keepdim=True)
        var = target.var(dim=-1, keepdim=True)
        target = (target - mean) / (var + 1.e-6) ** .5

    loss = (pred - target) ** 2
    loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
    loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches

    return loss

def DEC(model, dataset, config):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if cuda else "cpu")
    batch_size = 128
    
    #Graph regularization 
    # A=torch.Tensor(total)
    # d = 1.0/torch.sqrt(torch.abs(A.sum(axis=1)))
    # D_inv = torch.diag(d)
    # lap_mat =torch.diag(torch.sign(A.sum(axis=1)))-D_inv@A@D_inv
    
    awl = AutomaticWeightedLoss(3)
    optimizer = Adam([
            {'params': model.parameters()},
            {'params': awl.parameters(), 'weight_decay': 0}
        ], lr=config['lr'])
    data = dataset
    #y = dataset.y
    data = torch.Tensor(data).to(device)
    data=data.unsqueeze(1)
    
    if config['model'] == 'MAE':       
        total_samples = len(data)
        z_part = []
        q_part = []
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_data = data[i:i+batch_size]  # Get a batch of data
                batch_data = torch.Tensor(batch_data).to(device)
                #_, batch_result_z, _, _, batch_result_q = model(batch_data)
                _, batch_result_z, _, _, batch_result_q = model(batch_data)
                z_part.append(batch_result_z)
                q_part.append(batch_result_q)
        # Concatenate the results along the batch dimension
        z = torch.cat(z_part, dim=0)
        q = torch.cat(q_part, dim=0)

    #kmeans = KMeans(n_clusters=config['num_classes'], init='k-means++', n_init=10)
    #kmeans = KMeans(n_clusters=config['num_classes'], init='k-means++', n_init=10)
    X_np = z.cpu().detach().numpy()
    #y_pred = kmeans.fit_predict(X_np)
    kmeans = MiniBatchKMeans(n_clusters=config['num_classes'], batch_size=256)  # 调整batch_size以适应你的数据
    y_pred = kmeans.fit_predict(X_np)  # X是你的数据
    
    

    y_pred = np.array(y_pred)
    n_batch = data.size(0) // batch_size + 1
    total_samples = len(dataset)
    
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.train()
    
    # get the smm params
    alpha0 = 1.  # Assume an initial uniform distribution
    kappa0 = 0.0000
    rho0 = config['embed_dim_out'] + 2
    K = config['num_classes']
    
    # init SMM
    n_z = config['embed_dim_out']
    jmu = Parameter(torch.Tensor(K, n_z))
    torch.nn.init.xavier_normal_(jmu.data)
    jsig = Parameter(torch.Tensor(K, n_z, n_z))
    torch.nn.init.xavier_normal_(jsig.data)
    jpai = Parameter(torch.Tensor(K, 1))
    torch.nn.init.xavier_normal_(jpai.data)
    jv = Parameter(torch.Tensor(K, 1))
    torch.nn.init.xavier_normal_(jv.data)
    z = z.to('cpu')
    Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat, clusters = initialize_SMM_parameters(z, K, alpha0, kappa0, rho0)  
    
    # define the optimizer
    awl1 = AutomaticWeightedLoss(4)
    awl2 = AutomaticWeightedLoss(2)
    # for dataset_level
    optimizer1 = Adam([
                {'params': model.parameters()},
                {'params': awl1.parameters(), 'weight_decay': 0}
            ], lr=config['lr'])
    
    # for batch_level
    optimizer2 = Adam([
                {'params': model.parameters()},
                {'params': awl2.parameters(), 'weight_decay': 0}
            ], lr=1e-3*config['lr'])
    
    # for SMM
    optimizer3 = Adam([jmu, jsig, jpai, jv], lr=1e-5*config['lr'])
    y_pred_last = clusters
    
    for epoch in tqdm(range(30)):
        if epoch % config['interval'] == 0:     
            total_samples = len(dataset)
            x_bar_part = []
            z_part = []
            mask_part = []
            rloss_part = []
            with torch.no_grad():
                for i in range(0, total_samples, batch_size):
                    batch_data = data[i:i+batch_size]  # Get a batch of data
                    batch_data = torch.Tensor(batch_data).to(device)
                    batch_result_x_bar, batch_result_z, batch_result_rloss, batch_result_mask, _ = model(batch_data)

                    x_bar_part.append(batch_result_x_bar)
                    z_part.append(batch_result_z)
                    mask_part.append(batch_result_mask)
                    rloss_part.append(batch_result_rloss)
            # Concatenate the results along the batch dimension
            x_bar = torch.cat(x_bar_part, dim=0)
            z = torch.cat(z_part, dim=0)
            mask = torch.cat(mask_part, dim=0)
            rloss = caculate_rloss(data, x_bar, mask)
            
            Theta_prev, clusters, xi_i_k_history = EM_algorithm(z, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat,
                                                                S0_hat, rho0_hat, clusters,
                                                                max_iterations=5, config=config, tol=5 * 1e-3)
            
            q = xi_i_k_history[-1].data
            q = torch.where(q != 0., q, 1e-5) 
            q = q.to('cuda:1')
            #p = target_distribution(q)
            j = 0
            for theta in Theta_prev:
                theta_pai = torch.tensor(theta['pai']).to('cpu')  
                jpai.data[j] = torch.tensor(theta_pai.detach().numpy())
                j += 1
            jpai = jpai.to('cuda:1')
            likeli_loss = 0.01*likelihood(q, jpai)
            # lap_mat = lap_mat.to(device)
            # reg_loss=regularization(z,lap_mat)/(total_samples*100)
            size_loss = 0.01*size(q)
            #kl_loss = F.kl_div(q.log(), p)
            
            z = z.to(device)
            reconstr_loss = rloss
            
            # update encoder
            loss = awl1(reconstr_loss, size_loss, likeli_loss)
            #loss = (size_loss)

            loss.requires_grad_(True) 
            optimizer1.zero_grad()
            loss.backward(retain_graph=True)
            optimizer1.step()

            total_loss = loss.item()
            total_likeli_loss = likeli_loss.item()
            # total_reg_loss = reg_loss.item()
            total_reconstr_loss = reconstr_loss.item()
            total_size_loss = size_loss.item()
            # lap_mat = lap_mat.to('cpu')
            
            # print("epoch {} loss1={:.4f}".format(epoch,
            #                                     total_loss))
            # print('likeli_loss:', total_likeli_loss, 
            #       'reconstr_loss:', total_reconstr_loss,
            #      'size_loss:', total_size_loss)
            
            x_bar_part = []
            z_part = []
            mask_part = []
            with torch.no_grad():
                for i in range(0, total_samples, batch_size):
                    batch_data = data[i:i+batch_size]  # Get a batch of data
                    batch_data = torch.Tensor(batch_data).to(device)
                    batch_result_x_bar, batch_result_z, _, batch_result_mask, _ = model(batch_data)

                    x_bar_part.append(batch_result_x_bar)
                    z_part.append(batch_result_z)
                    mask_part.append(batch_result_mask)
            # Concatenate the results along the batch dimension
            x_bar = torch.cat(x_bar_part, dim=0)
            z = torch.cat(z_part, dim=0)
            Theta_prev, clusters, xi_i_k_history = EM_algorithm(z, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat,
                                                                S0_hat, rho0_hat, clusters,
                                                                max_iterations=0, config=config, tol=5 * 1e-3)
            
            q = torch.tensor(xi_i_k_history[-1])
            q = torch.where(q != 0., q, 1e-5)
            p = target_distribution(q)
            
            # evaluate clustering performance
            y_pred = clusters.cpu().detach().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            #print(len(np.unique(y_pred_last)))
            if epoch==9:
                break
            if epoch > 0 and delta_label < config['tol']:
                print('delta_label {:.4f}'.format(delta_label), '< tol', 10*config['tol'])
                print('Reached tolerance threshold. Stopping training.')
                break
        
        total_loss = 0.
        total_kl_loss = 0.
        total_reconstr_loss = 0.
        # total_reg_loss = 0.
        new_idx = torch.randperm(data.size()[0])
        
        for batch in range(n_batch):
            if batch < n_batch - 1:
                idx = new_idx[batch * batch_size:(batch + 1) * batch_size]
            else:
                break
                idx = new_idx[batch * batch_size:]

            x_train = data[idx, :, :, :]
            # lap_mat1=lap_mat[idx,:]
            # lap_mat1=lap_mat1[:,idx]
            
            x_bar, z, rloss, mask, _ = model(x_train)
            z = z.to('cpu')
           # _, _, xi_i_k_history = EM_algorithm(z, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat,
            #                                    clusters[idx], max_iterations=5, config=config, tol=5 * 1e-3)
            
            
            q = torch.tensor(xi_i_k_history[-1])
            q = torch.where(q != 0., q, 1e-5)
            q=q[idx]
            j = 0
            for theta in Theta_prev:
                theta_mu = torch.tensor(theta['mu']).to('cpu')  
                theta_sigma = torch.tensor(theta['sigma']).to('cpu')  
                theta_pai = torch.tensor(theta['pai']).to('cpu')  
                theta_v = torch.tensor(theta['v']).to('cpu')  
                jmu.data[j] = torch.tensor(theta_mu.detach().numpy())
                jsig.data[j] = torch.tensor(theta_sigma.detach().numpy())
                jpai.data[j] = torch.tensor(theta_pai.detach().numpy())
                jv.data[j] = torch.tensor(theta_v.detach().numpy())
                j += 1
            z = z.to(device)
            kl_loss = F.kl_div(q.log(), p[idx])
            # lap_mat1 = lap_mat1.to(device)
            # reg_loss=regularization(z,lap_mat1)
            reconstr_loss = rloss
            
            # update encoder
            loss = awl2(kl_loss, reconstr_loss)

            optimizer2.zero_grad()
            loss.backward(retain_graph=True)
            optimizer2.step()
            # update SMM
            loss1 = kl_loss.requires_grad_(True)
            optimizer3.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer3.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_reconstr_loss += reconstr_loss.item()
            # total_reg_loss += reg_loss.item()
            j = 0
            for theta in Theta_prev:
                theta['mu'] = jmu[j].data
                theta['sigma'] = torch.diag(torch.abs(torch.diag(jsig[j].data)))
                theta['pai'] = jpai[j].data
                theta['v'] = jv[j].data
                j += 1

        # print("epoch {} loss2={:.4f}".format(epoch,
        #                                     total_loss / (batch + 1)))
        # print('kl_loss:', total_kl_loss / (batch + 1), 
        #       'reconstr_loss:', total_reconstr_loss / (batch + 1),
        #      'reg_loss', total_reg_loss/(batch + 1))
    
    z_part = []
    with torch.no_grad():
        for i in range(0, total_samples, batch_size):
            batch_data = data[i:i+batch_size]  # Get a batch of data
            batch_data = torch.Tensor(batch_data).to(device)
            _, batch_result_z, _, _, _ = model(batch_data)
            z_part.append(batch_result_z)
                   
    # Concatenate the results along the batch dimension
    z = torch.cat(z_part, dim=0)
    #xi_i_k = calculate_xi(z, Theta_prev)
    #y_pred_last = torch.argmax(xi_i_k, dim=1).cpu().numpy()  
    Theta_prev, clusters, xi_i_k_history = EM_algorithm(z, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat, clusters,max_iterations=0, config=config, tol=5 * 1e-3)
    y_pred_last = clusters
    
    return z, model
