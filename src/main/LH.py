import torch
import numpy as np
from sympy.stats import Gamma
from torch.distributions import MultivariateNormal
import torch.distributions.multivariate_normal as mvn
from sklearn.cluster import KMeans


def calculate_S_mle_k(X, p_ik, mu_k):
    mu_k_reshaped = mu_k.view(-1, X.shape[1])
    X_minus_mu = X - mu_k_reshaped
    S_mle_k = torch.mm(X_minus_mu.t(), X_minus_mu * p_ik.unsqueeze(1)) / torch.sum(p_ik)
    return S_mle_k


def calculate_xi(X, Theta_updated):
    K = len(Theta_updated)

    xi = torch.zeros((len(X), K))
    for j in range(K):
        # print(Theta_updated[j]['sigma'])
        dist = mvn.MultivariateNormal(Theta_updated[j]['mu'], Theta_updated[j]['sigma'])
        xi[:, j] = Theta_updated[j]['pai'] * dist.log_prob(X).exp()
    xi = torch.where(xi != torch.inf, xi, torch.ones((len(X), K)))
    xi = torch.where(torch.isnan(xi), torch.zeros((len(X), K)), xi)
    xi = torch.abs(xi)
    xi_sum = torch.sum(xi, dim=1)
    xi = (xi + 1e-6) / (xi_sum.unsqueeze(1) + 1e-6)

    return xi


def initialize_SMM_parameters(X, K, alpha0, kappa0, rho0):
    # Convert samples to NumPy array for K-means clustering
    X_np = X.detach().numpy()

    # Perform K-means clustering on the data
    kmeans = KMeans(n_clusters=K, n_init=20)
    cluster_labels = kmeans.fit_predict(X_np)

    # Convert cluster labels back to PyTorch tensor
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
    D = X.shape[1]
    Theta = []

    # Convert K-means estimated parameters to the desired format
    for k in range(K):
        theta_k = {}
        theta_k['pai'] = torch.tensor(1.0 / K)
        theta_k['mu'] = torch.tensor(kmeans.cluster_centers_[k], dtype=torch.float32)

        cluster_samples = X[cluster_labels == k]
        theta_k['sigma'] = torch.diag(
            torch.diag(torch.tensor(torch.cov(cluster_samples.T), dtype=torch.float32)) + 1e-6)
        # theta_k['sigma']=torch.where(theta_k['sigma']!=torch.nan,theta_k['sigma'],torch.eye(D))
        if cluster_samples.size(0) < 2:
            theta_k['sigma'] = torch.eye(D) * 1e-6
        theta_k['v'] = torch.tensor(1.0 / K)
        Theta.append(theta_k)

    m0 = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32).mean()
    S0 = torch.mean(torch.stack([torch.diag(theta['sigma']) for theta in Theta]), dim=0)

    alpha0_hat = alpha0 * torch.ones(K)
    m0_hat = (kappa0 * m0 + K * torch.mean(torch.stack([theta['mu'] for theta in Theta]), dim=0)) / (kappa0 + K)
    kappa0_hat = kappa0 + K
    # S0_hat = S0 + torch.sum(torch.stack([theta['sigma'] + kappa0 * torch.outer(theta['mu'] - m0_hat, theta['mu'] - m0_hat) for theta in Theta]), dim=0)
    # Calculate S_mle_k for each cluster
    S_mle_k_list = []
    for k in range(K):
        xi_i_k = calculate_xi(X, Theta)

        p_ik = xi_i_k[:, k]
        mu_k = Theta[k]['mu']
        S_mle_k = calculate_S_mle_k(X, p_ik, mu_k)
        S_mle_k_list.append(S_mle_k)

    # Update S0_hat using S_mle_k
    S0_hat = S0 + torch.sum(torch.stack(S_mle_k_list), dim=0)
    rho0_hat = rho0 + K

    return Theta, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat, cluster_labels


import torch
from torch.distributions import MultivariateNormal


def update_SMM_parameters(X, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat):
    K = len(Theta_prev)
    D = X.shape[1]

    xi_i_k = calculate_xi(X, Theta_prev)
    # alpha_k =torch.distributions.dirichlet.Dirichlet(alpha0_hat).rsample()
    alpha_k = calculate_alpha_k(xi_i_k, alpha0_hat)
    alpha_k = (alpha_k - 1) / (alpha_k.sum() - K + 1e-6)
    for k in range(K):

        p_ik = xi_i_k[:, k]

        N_k = torch.sum(p_ik)
        if torch.isnan(N_k):
            N_k = 0
        if torch.isinf(N_k):
            N_k = 1
        x_bar_k = torch.matmul(p_ik, X) / (N_k + 1e-6)

        m_k = (kappa0_hat * m0_hat + N_k * x_bar_k) / (kappa0_hat + N_k)
        kappa_k = kappa0_hat + N_k
        S_k = S0_hat + torch.matmul(p_ik * (X - x_bar_k).T, X - x_bar_k) + \
              (kappa0_hat * N_k) / (kappa0_hat + N_k) * torch.outer(x_bar_k - m0_hat, x_bar_k - m0_hat)
        rho_k = rho0_hat + N_k

        if torch.isnan(S_k).any() or torch.isinf(S_k).any():
            continue

        Theta_prev[k]['mu'] = m_k
        Theta_prev[k]['sigma'] = S_k / rho_k
        Theta_prev[k]['sigma'] = torch.abs(torch.diag(torch.diag(Theta_prev[k]['sigma']) + 1e-6))
        Theta_prev[k]['v'] = D + 1 + N_k
        Theta_prev[k]['pai'] = alpha_k[k]

    return Theta_prev


def estimate_initial_sigma(X):
    X_mean = torch.mean(X, dim=0, keepdim=True)
    X_centered = X - X_mean
    cov_matrix = torch.matmul(X_centered.t(), X_centered)
    initial_sigma = torch.mean(torch.diagonal(cov_matrix))
    # return initial_sigma.item()
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


def calculate_sigma(X, p_ik, mu_k):
    K = mu_k.shape[0]

    sigma_k_list = torch.ones((K, mu_k.shape[1], mu_k.shape[1]))

    for k in range(K):
        mu_k_reshaped = mu_k[k].view(-1, X.shape[1])
        X_minus_mu = X - mu_k_reshaped
        sigma_k_list[k] = torch.mm(X_minus_mu.t(), X_minus_mu * p_ik[:, k].unsqueeze(1)) / torch.sum(p_ik[:, k])

    return sigma_k_list


def calculate_v(X, p_ik):
    v_k = torch.sum(p_ik, dim=0) / len(X)
    return v_k


# def calculate_pai(xi_i_k, p_ik):
#   pai_k = torch.sum(p_ik, dim=0) / len(xi_i_k)
#  return pai_k
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

