import torch
import math
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from torch.optim import Adam
from torch.nn.parameter import Parameter
from SMM import initialize_SMM_parameters
from EM import EM_algorithm
from LH import likelihood, regularization, size
from torch.utils.data import DataLoader



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
    ind = (np.array(linear_assignment(w.max() - w))).transpose()
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size


def DEC(model, dataset, total, config):

    #  model.pretrain('data/ae_mnist.pkl')
    #model.pretrain(path='')

    batch_size = config['batch_size']
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Graph regularization
    A = torch.Tensor(total).to(device)
    d = 1.0 / torch.sqrt(torch.abs(A.sum(axis=1)))
    D_inv = torch.diag(d)
    lap_mat = torch.diag(torch.sign(A.sum(axis=1))) - D_inv @ A @ D_inv

    # cluster parameter initiate
    data = dataset

    data = torch.Tensor(data).to(device)
    n_batch = data.size(0) // batch_size + 1
    data = data.unsqueeze(1)
    x_bar, z, rloss = model(data)


    model.train()
    # Set the remaining variables based on prior knowledge or assumptions
    alpha0 = 1.
    kappa0 = 0.0000
    rho0 = config['embed_dim'] + 2
    K = config['num_classes']
    x_bar, z, rloss = model(data)
    n_z = config['embed_dim']
    jmu = Parameter(torch.Tensor(K, n_z))
    torch.nn.init.xavier_normal_(jmu.data)
    jsig = Parameter(torch.Tensor(K, n_z, n_z))
    torch.nn.init.xavier_normal_(jsig.data)
    jpai = Parameter(torch.Tensor(K, 1))
    torch.nn.init.xavier_normal_(jpai.data)
    jv = Parameter(torch.Tensor(K, 1))
    torch.nn.init.xavier_normal_(jv.data)
    z = z.to('cpu')
    Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat, clusters = initialize_SMM_parameters(z, K, alpha0,
                                                                                                       kappa0, rho0)
    y_pred_last = clusters

    optimizer = Adam(model.parameters(), lr=config['lr'])
    optimizer1 = Adam([jmu, jsig, jpai, jv], lr=config['lr'])
    with torch.autograd.detect_anomaly():
        # if 1:
        for epoch in range(0):
            total_loss = 0.

            x_bar, z, rloss = model(data)
            Theta_updated, clusters, xi_i_k_history = EM_algorithm(z, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat,
                                                                   S0_hat, rho0_hat, clusters,
                                                                   max_iterations=5,  config=config, tol=5 * 1e-3)

            q = torch.tensor(xi_i_k_history[-1])
            j = 0
            for theta in Theta_updated:
                jpai.data[j] = torch.tensor(theta['pai'].detach().numpy())
                j += 1

            # evaluate clustering performance
            y_pred = clusters
            delta_label = torch.sum(y_pred != y_pred_last) / y_pred.size(0)
            y_pred_last = y_pred
            if epoch > 0 and delta_label < config['tol']:
                print('delta_label {:.4f}'.format(delta_label), '< tol', config['tol'])
                print('Reached tolerance threshold. Stopping training.')
                break

            likeli_loss = likelihood(q, jpai)

            reg_loss = regularization(z, lap_mat)
            size_loss = size(q)

            reconstr_loss = rloss
            # update encoder
            loss = - config['l1'] * likeli_loss + config['l3'] * math.e ** (
                        -epoch / 3) * reg_loss - config['l4'] * size_loss + config['l6'] * reconstr_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()

            total_loss = loss.item()
            total_likeli_loss = likeli_loss.item()

            total_reg_loss = reg_loss.item()
            total_size_loss = size_loss.item()

            total_reconstr_loss = reconstr_loss.item()
            print("epoch {} loss={:.4f}".format(epoch + 1,
                                                total_loss))
            print('likeli_loss:', total_likeli_loss, 'reg_loss:', total_reg_loss, 'size_loss:', total_size_loss,
                  'reconstr_loss:', total_reconstr_loss)
            Theta_prev = Theta_updated

    for epoch in range(0):
        # with torch.autograd.detect_anomaly():
        if epoch % config['interval'] == 0:

            x_bar, z, rloss = model(data)
            Theta_prev, clusters, xi_i_k_history = EM_algorithm(z, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat,
                                                                S0_hat, rho0_hat, clusters,
                                                                max_iterations=5, config=config, tol=5 * 1e-3)

            # update target distribution p
            tmp_q = xi_i_k_history[-1].data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = clusters.cpu().detach().numpy()
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            if epoch > 0 and delta_label < config['tol']:
                print('delta_label {:.4f}'.format(delta_label), '< tol', config['tol'])
                print('Reached tolerance threshold. Stopping training.')
                break

        total_loss = 0.
        total_kl_loss = 0.
        total_reconstr_loss = 0.
        total_reg_loss = 0.

        new_idx = torch.randperm(data.size()[0])
        for batch in range(n_batch):
            if batch < n_batch - 1:
                idx = new_idx[batch * batch_size:(batch + 1) * batch_size]
            else:
                idx = new_idx[batch * batch_size:]

            x_train = data[idx, :, :, :]

            lap_mat1 = lap_mat[idx, :]
            lap_mat1 = lap_mat1[:, idx]

            x_bar, z, rloss = model(x_train)
            _, _, xi_i_k_history = EM_algorithm(z, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat,
                                                clusters[idx], max_iterations=0, config=config, tol=5 * 1e-3)

            q = torch.tensor(xi_i_k_history[0])

            j = 0
            for theta in Theta_prev:
                jmu.data[j] = torch.tensor(theta['mu'].detach().numpy())
                jsig.data[j] = torch.tensor(theta['sigma'].detach().numpy())
                jpai.data[j] = torch.tensor(theta['pai'].detach().numpy())
                jv.data[j] = torch.tensor(theta['v'].detach().numpy())
                j += 1

            kl_loss = F.kl_div(q.log(), p[idx])
            reconstr_loss = rloss
            reg_loss = regularization(z,lap_mat1)
            # update encoder
            loss = config['gamma'] * kl_loss + config['l6'] * reconstr_loss + config['l3'] * math.e ** (-(epoch + 10) / 3) * reg_loss

            optimizer.zero_grad()
            loss.backward(retain_graph=True)
            optimizer.step()
            # update SMM
            loss1 = kl_loss.requires_grad_(True)

            optimizer1.zero_grad()
            loss1.backward(retain_graph=True)
            optimizer1.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_reconstr_loss += reconstr_loss.item()
            total_reg_loss += reg_loss.item()
            j = 0
            for theta in Theta_prev:
                theta['mu'] = jmu[j].data
                theta['sigma'] = torch.diag(torch.abs(torch.diag(jsig[j].data)))
                theta['pai'] = jpai[j].data
                theta['v'] = jv[j].data
                j += 1

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch + 1)))
        print('kl_loss:', total_kl_loss / (batch + 1), 'reconstr_loss:', total_reconstr_loss / (batch + 1), 'reg_loss:',
              total_reg_loss / (batch + 1))
    return y_pred_last
