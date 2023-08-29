import torch
import time
import numpy as np
from sympy.stats import Gamma
from sklearn.cluster import KMeans
from torch.distributions import MultivariateNormal
import torch.distributions.multivariate_normal as mvn


def calculate_S_mle_k(X, Omega_ik, x_k):
    N = X.shape[0]
    D = X.shape[1]
    S_mle_k = torch.ones(D, D).to('cuda')
    x_k_reshaped = x_k.view(-1, X.shape[1])
    X_minus_mu = X - x_k_reshaped
    # the weighted average sum of the mean-centered covariance matrix
    S_mle_k = torch.mm(X_minus_mu.t(), X_minus_mu)
    return S_mle_k


def calculate_xi(X, Theta_updated):
    K = len(Theta_updated)  # the number of clusters
    xi = torch.zeros((X.shape[0], K)).to('cpu')  # init xi[N, K]
    X_np = X.to('cpu')
    for j in range(K):
        theta_j_mu = torch.tensor(Theta_updated[j]['mu']).to('cpu')
        theta_j_sigma = torch.tensor(Theta_updated[j]['sigma']).to('cpu')
        theta_j_pai = torch.tensor(Theta_updated[j]['pai']).to('cpu')
        dist = mvn.MultivariateNormal(theta_j_mu, theta_j_sigma)  # creat phi
        xi[:, j] = theta_j_pai * dist.log_prob(X_np).exp()
    xi = torch.where(xi != torch.inf, xi, torch.ones((len(X), K)))  # outlier process
    xi = torch.where(torch.isnan(xi), torch.zeros((len(X), K)), xi)  # outlier process
    xi = torch.abs(xi).to('cuda')
    xi_sum = torch.sum(xi, dim=1)
    xi = (xi + 1e-6) / (xi_sum.unsqueeze(1) + 1e-6)

    return xi


def calaculate_sigma(X, Theta_updated):
    K = len(Theta_updated)
    D = X.shape[1]
    sigma = torch.zeros(X.shape[0], K).to('cuda')
    for k in range(K):
        theta_k_mu = torch.tensor(Theta_updated[k]['mu']).unsqueeze(dim=0).to('cuda')
        theta_k_sigma = torch.tensor(Theta_updated[k]['sigma']).to('cuda')
        sigma[:, k] = torch.diag(torch.mm(torch.mm(X-theta_k_mu, torch.inverse(theta_k_sigma)), (X-theta_k_mu).t()))
    
    return sigma


def calculate_zeta(X, Theta_updated, sigma):
    D = X.shape[1]
    K = len(Theta_updated)
    zeta = torch.zeros(X.shape[0], K).to('cuda')
    for k in range(K):
        theta_k_v = torch.tensor(Theta_updated[k]['v']).to('cuda')
        zeta[:, k] = (theta_k_v+D)/(theta_k_v+sigma[:, k])
    
    return zeta


def calculate_Omegacalculate_Omega(X, Theta_prev):
    xi = calculate_xi(X, Theta_updated)
    sigma = calaculate_sigma(X, Theta_updated)
    zeta = calculate_zeta(X, Theta_updated, sigma)
    Omega = torch.mul(xi, zeta)
    
    return Omega
    

def calculate_x_k(X, k, Omega):
    Omega_k = torch.sum(Omega[:, k])
    tem1 = torch.ones(1, X.shape[1])
    for i in range(X.shape[0]):
        tem2 = torch.ones(1, X.shape[1])*Omega[i, k]
        tem1 = torch.cat((tem1, tem2), dim=0)
    tem1 = tem1[1:, :].sum(dim=0)
    x_k = tem1/Omega_k
    
    return x_k


def caculate_v_k(v, k, xi, zeta):
    v = torch.tensor(v).to('cuda')
    xi_k = xi[:, k]
    zeta_k = zeta[:, k]
    
    for i in range(3):
        der = torch.tensor(0.).to('cuda')
        der = torch.sum(xi_k*(0.5*torch.log(v/2).item()+0.5-0.5*torch.digamma(v/2).item()+0.5*(torch.log(zeta_k)-zeta_k)), dim=0)
        v = v - 0.001*der
        
    return v
    
    
    
def initialize_SMM_parameters(X, K, alpha0, kappa0, rho0):
     #Convert samples to NumPy array for K-means clustering
    X_np = X.detach().numpy()

    # Perform K-means clustering on the data
    kmeans = KMeans(n_clusters=K, n_init=20)
    cluster_labels = kmeans.fit_predict(X_np)

    # Convert cluster labels back to PyTorch tensor
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
    D = X.shape[1]
    N = X.shape[0]
    Theta = []

    # Convert K-means estimated parameters to the desired format
    for k in range(K):
        theta_k = {}
        theta_k['pai'] = torch.tensor(1.0 / K)
        theta_k['mu'] = torch.tensor(kmeans.cluster_centers_[k], dtype=torch.float32)

        cluster_samples = X[cluster_labels == k]  # [N_cluste, D]
        theta_k['sigma'] = torch.diag(
            torch.diag(torch.tensor(torch.cov(cluster_samples.T), dtype=torch.float32)) + 1e-6)
        if cluster_samples.size(0) < 2:
            theta_k['sigma'] = torch.eye(D) * 1e-6
        theta_k['v'] = torch.tensor(3.0)
        Theta.append(theta_k)

    m0 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).mean()
    S0 = K**(-1/D)*torch.diag(torch.diag(torch.mm((X-X/N).t(), X-X/N)) + 1e-6)
    S0=torch.mean(torch.stack([torch.diag(theta['sigma'])  for theta in Theta]),dim=0)

    alpha0_hat = alpha0 * torch.ones(K)
    m0_hat = m0 
    kappa0_hat = kappa0
    S0_hat = S0
    rho0_hat = rho0

    return Theta, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat, cluster_labels



def update_SMM_parameters(X, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat):
    K = len(Theta_prev)
    D = X.shape[1]
    
    sigma = calaculate_sigma(X, Theta_prev)  # to cuda
    xi_i_k = calculate_xi(X, Theta_prev)  # to cuda
    zeta_i_k = calculate_zeta(X, Theta_prev, sigma)    # to cuda

    alpha_k = calculate_alpha_k(xi_i_k, alpha0_hat)  # to cuda
    alpha_k = (alpha_k - 1) / (alpha_k.sum() - K + 1e-6)
    Omega = torch.mul(xi_i_k, zeta_i_k)  # to cuda
    for k in range(K):
        p_ik = xi_i_k[:, k] 
        q_ik = zeta_i_k[:, k]
        N_ik = p_ik*q_ik       
        N_k = torch.sum(p_ik*q_ik)
        
        if np.isnan(N_k.item()):
            N_k = torch.tensor(0.).to('cuda')    
            
        x_bar_k = torch.matmul(N_ik, X) / (N_k + 1e-6)
        mask = torch.isnan(x_bar_k)
        
        if mask.sum().item()!=0:
            x_bar_k = X.mean(dim=0)
            
        kappa_k = kappa0_hat + N_k
        m_k = (kappa0_hat * m0_hat + N_k * x_bar_k) / (kappa_k)
        
        if kappa_k == 0:
            m_k = x_bar_k
        
        Omega_ik = Omega[:, k]              
        S_mle_k = calculate_S_mle_k(X, Omega_ik, x_bar_k)

        
        
        S_k = S0_hat + S_mle_k
        mask = torch.isnan(S_k)
        if mask.sum().item()!=0:
            S_k = S0_hat
            
        if np.isnan(torch.sum(p_ik).item()):
            rho_k = rho0_hat
        else:
            rho_k = rho0_hat + torch.sum(p_ik)
        v = Theta_prev[k]['v']
        Theta_prev[k]['mu'] = m_k
        Theta_prev[k]['sigma'] = S_k / (rho_k+D+2)           
        Theta_prev[k]['sigma'] = torch.abs(torch.diag(torch.diag(Theta_prev[k]['sigma']) + 1e-6))       
        Theta_prev[k]['v'] = caculate_v_k(Theta_prev[k]['v'], k, xi_i_k, zeta_i_k)
        Theta_prev[k]['pai'] = alpha_k[k]
    return Theta_prev



def estimate_initial_sigma(X):
    X_mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - X_mean
    cov_matrix = torch.matmul(X_centered.t(), X_centered)
    initial_sigma = torch.mean(torch.diagonal(cov_matrix))
    
    return initial_sigma


def calculate_log_Gamma(X, xi_i_k, Theta):
    K = len(Theta)
    log_Gamma = []

    for k in range(K):
        mu_k = Theta[k]['mu']
        sigma_k = Theta[k]['sigma']
        v_k = Theta[k]['v']

        log_gamma_k = -0.5 * torch.logdet(2 * np.pi * sigma_k) - 0.5 * torch.matmul((X - mu_k.reshape(1, -1)),
                                                                                    torch.inverse(sigma_k)).matmul(
            (X - mu_k.reshape(1, -1)).T) + torch.log(v_k * xi_i_k[:, k].mean())
        log_Gamma.append(log_gamma_k)

    return torch.stack(log_Gamma)


def calculate_mu(X, p_ik):
    p_ik_t = p_ik.t()
    numerator = torch.mm(p_ik_t, X)

    denominator = torch.sum(p_ik, dim=0, keepdim=True) + 1e-6

    mu = numerator / denominator.t()
    return mu


def calculate_v(X, p_ik):
    v_k = torch.sum(p_ik, dim=0) / len(X)
    return v_k


def calculate_alpha_k(xi_i_k, alpha0_hat):
    alpha_k = alpha0_hat + torch.sum(xi_i_k, dim=0)
    return alpha_k


def calculate_z(X, v_kt, sigma_kt):
    D = X.shape[1]
    gamma_dist = Gamma(v_kt + D / 2, 1 / (v_kt + sigma_kt / 2))
    z_i_kt = gamma_dist.sample()
    log_z_i_kt = gamma_dist.log_prob(z_i_kt)
    return z_i_kt, log_z_i_kt


def assign_clusters(X, Theta_updated):
    xi_i_k = calculate_xi(X, Theta_updated)
    clusters = torch.argmax(xi_i_k, dim=1)
    return clusters

