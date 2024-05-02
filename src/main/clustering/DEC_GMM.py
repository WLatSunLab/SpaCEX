import argparse
import torch
import numpy as np
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from sklearn.mixture import GaussianMixture
from SpaCEX.src.main.LH import likelihood, regularization, size
from torch.nn import Linear


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

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

def cluster_acc(y_true, y_pred):
    """
    Calculate clustering accuracy. Require scikit-learn installed
    # Arguments
        y: true labels, numpy.array with shape `(n_samples,)`
        y_pred: predicted labels, numpy.array with shape `(n_samples,)`
    # Return
        accuracy, in [0,1]
    """
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    #from sklearn.utils.linear_assignment_ import linear_assignment
    from scipy.optimize import linear_sum_assignment as linear_assignment
    ind = (np.array(linear_assignment(w.max() - w))).transpose()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def DEC(model, dataset, total, config):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda:1" if cuda else "cpu")
    batch_size = 64
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

    kmeans = KMeans(n_clusters=config['num_classes'], init='k-means++', n_init=10)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    
    y_pred_count = torch.tensor(y_pred)
    counts = torch.bincount(y_pred_count)
    y_pred = np.array(y_pred)
    n_batch = data.size(0) // batch_size + 1
    total_samples = len(dataset)
    z = None
    
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.train()
    for epoch in tqdm(range(30), desc="Clustering"):
        if epoch % config['interval'] == 0:
            if config['model'] == 'MAE':       
                total_samples = len(data)
                tmp_q_part = []
                z_part = []
                with torch.no_grad():
                    for i in range(0, total_samples, batch_size):
                        batch_data = data[i:i+batch_size]  # Get a batch of data
                        batch_data = torch.Tensor(batch_data).to(device)
                        _, batch_result_z, _, _, batch_result_tmp_q = model(batch_data)
                        tmp_q_part.append(batch_result_tmp_q)
                        z_part.append(batch_result_z)
                tmp_q = torch.cat(tmp_q_part, dim=0)
                z = torch.cat(z_part, dim=0)
            # update target distribution p
            X = z.detach().cpu().numpy()
            gmm = GaussianMixture(n_components=2, random_state=42)

            # 拟合模型
            gmm.fit(X)

            # 获取每个样本的后验概率
            tem_q = gmm.predict_proba(X)
            tem_q = torch.tensor(tem_q).to('cuda')
            p = target_distribution(tmp_q)
            
            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred
            y_pred_count = torch.tensor(y_pred_last)
            counts = torch.bincount(y_pred_count)

            if epoch > 0 and delta_label < config['tol']:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      config['tol'])
                print('Reached tolerance threshold. Stopping training.')
                break
        new_idx = torch.randperm(data.size()[0])
        
        total_kl_loss = 0.
        total_reconstr_loss = 0.
        total_size_loss = 0.
        for batch in range(n_batch):
            
            if batch < n_batch - 1:
                idx = new_idx[batch * batch_size:(batch + 1) * batch_size]
            else:
                idx = new_idx[batch * batch_size:]
                
            x_train = data[idx, :, :, :]
            
            x_bar, z, reconstr_loss, _, q = model(x_train)
            kl_loss = F.kl_div(q.log(), p[idx])
            size_loss = size(q)
            loss = awl(kl_loss, reconstr_loss)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_reconstr_loss = total_reconstr_loss + reconstr_loss.item()
            total_kl_loss = total_kl_loss + kl_loss.item()
            total_size_loss = total_size_loss + size_loss.item()
    
    return y_pred_last, z, model
