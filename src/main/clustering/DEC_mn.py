import argparse
import torch
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


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


def DEC(model, dataset, config):
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    batch_size = config['batch_size']
    train_loader = DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=False)
    optimizer = Adam(model.parameters(), lr=config['lr'])
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    data=data.unsqueeze(1)
    
    if config['model'] == 'MAE':       
        total_samples = len(data)
        z_part = []
        with torch.no_grad():
            for i in range(0, total_samples, batch_size):
                batch_data = data[i:i+batch_size]  # Get a batch of data
                batch_data = torch.Tensor(batch_data).to(device)
                _, batch_result_z, _, _, _ = model(batch_data)
                z_part.append(batch_result_z)
        # Concatenate the results along the batch dimension
        z = torch.cat(z_part, dim=0)

    kmeans = KMeans(n_clusters=config['num_classes'], n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    acc = cluster_acc(y, y_pred)
    print("acc ={:.4f}".format(acc))
    
    z = None
    
    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    model.train()
    for epoch in range(10):
        if epoch % config['interval'] == 0:
            if config['model'] == 'MAE':       
                total_samples = len(data)
                tmp_q_part = []
                with torch.no_grad():
                    for i in range(0, total_samples, batch_size):
                        batch_data = data[i:i+batch_size]  # Get a batch of data
                        batch_data = torch.Tensor(batch_data).to(device)
                        _, _, _, _, batch_result_tmp_q = model(batch_data)
                        tmp_q_part.append(batch_result_tmp_q)
                # Concatenate the results along the batch dimension
                tmp_q = torch.cat(tmp_q_part, dim=0)
            
            # update target distribution p
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            print('Iter {}'.format(epoch), ':Acc {:.4f}'.format(acc),
                  ', nmi {:.4f}'.format(nmi), ', ari {:.4f}'.format(ari))

            if epoch > 0 and delta_label < config['tol']:
                print('delta_label {:.4f}'.format(delta_label), '< tol',
                      config['tol'])
                print('Reached tolerance threshold. Stopping training.')
                break
        for batch_idx, (x, _, idx) in enumerate(train_loader):

            x = x.to(device)
            idx = idx.to(device)

            x_bar, z, reconstr_loss, _, q = model(x)

            #reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = config['gamma'] * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    return y_pred_last
