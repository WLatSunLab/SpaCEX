import torch
import time
import numpy as np
from sympy.stats import Gamma
from sklearn.cluster import KMeans
from torch.distributions import MultivariateNormal
import torch.distributions.multivariate_normal as mvn
#from scipy.linalg import svd
import torch.linalg as la


device = torch.device("cuda:1")
def calculate_S_mle_k(X, Omega_ik, x_k):
    N = X.shape[0]
    D = X.shape[1]
    S_mle_k = torch.ones(D, D)
    #.to('cuda')
    x_k_reshaped = x_k.view(-1, X.shape[1])
    X_minus_mu = X - x_k_reshaped
    # the weighted average sum of the mean-centered covariance matrix
    S_mle_k = torch.mm(torch.mul(Omega_ik.unsqueeze(dim=0).repeat(D, 1), X_minus_mu.t()), X_minus_mu)
    return S_mle_k

def isPD(B):
    """Returns true when input is positive-definite, via Cholesky"""
    try:
        _ = la.cholesky(B)
        return True
    except la.LinAlgError:
        return False
def nearestPD(A):
    """Find the nearest positive-definite matrix to input

    A Python/Numpy port of John D'Errico's `nearestSPD` MATLAB code [1], which
    credits [2].

    [1] https://www.mathworks.com/matlabcentral/fileexchange/42885-nearestspd

    [2] N.J. Higham, "Computing a nearest symmetric positive semidefinite
    matrix" (1988): https://doi.org/10.1016/0024-3795(88)90223-6
    """
    


    #H = np.dot(V.T, np.dot(np.diag(s), V))
    
    A = torch.tensor(A).to(device)
    B = (A + A.T) / 2
    _, s, V = la.svd(B)
    
    s = torch.tensor(s).to(device)
    V = torch.tensor(V).to(device)
    q = torch.diag(s)*V
    H = V.T*q
    #H = V.T@torch.diag(s)@V
    A2 = (B + H) / 2

    A3 = (A2 + A2.T) / 2

    if isPD(A3):
        return A3+1e-4*torch.eye(A3.size(0)).to(device)
    
    spacing = np.spacing(la.norm(A).cpu())
    # The above is different from [1]. It appears that MATLAB's `chol` Cholesky
    # decomposition will accept matrixes with exactly 0-eigenvalue, whereas
    # Numpy's will not. So where [1] uses `eps(mineig)` (where `eps` is Matlab
    # for `np.spacing`), we use the above definition. CAVEAT: our `spacing`
    # will be much larger than [1]'s `eps(mineig)`, since `mineig` is usually on
    # the order of 1e-16, and `eps(1e-16)` is on the order of 1e-34, whereas
    # `spacing` will, for Gaussian random matrixes of small dimension, be on
    # othe order of 1e-16. In practice, both ways converge, as the unit test
    # below suggests.
    I = torch.eye(A.size(0)).to(device)
    k = 1
    while not isPD(A3):
        mineig = torch.min(torch.real(la.eigvals(A3)))
        A3 += I * (-mineig * k**2 + torch.tensor(spacing).to(device))
        k += 1

    return A3+1e-4*I

# ok
def calculate_xi(X, Theta_updated,clusters):  
    K = len(Theta_updated)  # the number of clusters
    xi = torch.zeros((X.shape[0], K)).to(device)  # init xi[N, K]#####
    X_np = X.to(device) ##########
    nk=torch.zeros((1,K)).to(device) 
    for j in range(K):
        nk[:,j]=torch.sum(clusters==j)
        theta_j_mu = torch.tensor(Theta_updated[j]['mu']).to(device)###########
        theta_j_sigma = torch.tensor(Theta_updated[j]['sigma']).to(device)###############
         
        #theta_j_sigma=nearestPD(theta_j_sigma)
        #theta_j_sigma = torch.tensor(theta_j_sigma).to(device)

        theta_j_pai = torch.tensor(Theta_updated[j]['pai']).to(device)###############
        dist = MultivariateNormal(theta_j_mu, theta_j_sigma)  # creat phi
      
        xi[:, j] = theta_j_pai * dist.log_prob(X_np).exp()
        '''
        if torch.sum(torch.isnan(xi[:,j]))>0:
            print(j)
            print(theta_j_mu)
            print(theta_j_sigma)
            print(X_np[:10])
    '''
    xi = torch.where(xi != torch.inf, xi, 1e1*torch.ones((len(X), K)).to(device))  # outlier process
    xi = torch.where(torch.isnan(xi), 1e-6*torch.ones((len(X), K)).to(device), xi)  # outlier process torch.rand(1).to(device)*
    xi=nk*xi
    xi_sum = torch.sum(xi, dim=1)+1e-6

    xi = (xi) / (xi_sum.unsqueeze(1))
    return xi


def calaculate_sigma(X, Theta_updated):
    K = len(Theta_updated)
    D = X.shape[1]
    sigma = torch.zeros(X.shape[0], K).to(device)
   
    for k in range(K):
        theta_k_mu = torch.tensor(Theta_updated[k]['mu']).unsqueeze(dim=0).to(device)
        
        theta_k_sigma = torch.tensor(Theta_updated[k]['sigma']).to(device)
          # theta_k_mu.shape [1, 16]
        sigma[:, k] = torch.diag(torch.mm(torch.mm(X-theta_k_mu, torch.inverse(theta_k_sigma)), (X-theta_k_mu).t()))   
        
    return sigma


def calculate_zeta(X, Theta_updated, sigma):
    D = X.shape[1]
    K = len(Theta_updated)
    zeta = torch.zeros(X.shape[0], K).to(device)
    for k in range(K):
        theta_k_v = torch.tensor(Theta_updated[k]['v']).to(device)
        zeta[:, k] = (theta_k_v+D)/(theta_k_v+sigma[:, k])
    
    return zeta


def calculate_alpha_k(xi_i_k, alpha0_hat):
    alpha_k = alpha0_hat + torch.sum(xi_i_k, dim=0)
    return alpha_k


def caculate_v_k(v, k, xi, zeta):
    v = torch.tensor(v).to(device)
   
    xi_k = xi[:, k]
    zeta_k = zeta[:, k]
    
    for i in range(3):
        der = torch.tensor(0.).to(device)
        der = torch.sum(xi_k*(0.5*torch.log(v/2).item()+0.5-0.5*torch.digamma(v/2).item()+0.5*(torch.log(zeta_k)-zeta_k)), dim=0)
        v_new = v - 0.0001*der
        if v_new<=0:
            v_new = v

        
    return v_new
    
    
from sklearn.cluster import MiniBatchKMeans
def initialize_SMM_parameters(X, K, alpha0, kappa0, rho0):
     #Convert samples to NumPy array for K-means clustering
    X_np = X.detach().numpy()

    kmeans = MiniBatchKMeans(n_clusters=K, batch_size=256)
    cluster_labels = kmeans.fit_predict(X_np)  # X是你的数据

    # Convert cluster labels back to PyTorch tensor
    cluster_labels = torch.tensor(cluster_labels, dtype=torch.long)
    D = X.shape[1]
    N = X.shape[0]
    Theta = []

    # Convert K-means estimated parameters to the desired format
    for k in range(K):
        theta_k = {}
        cluster_samples = X[cluster_labels == k]  # [N_cluste, D]
        
        theta_k['mu'] = torch.tensor(kmeans.cluster_centers_[k], dtype=torch.float32)  # [N, 1]
        
        if torch.sum(torch.isnan(theta_k['mu']))>0:
            print(k)

        
        theta_k['pai'] = torch.tensor(1.0 / K)
        theta_k['sigma'] = torch.diag(torch.diag(torch.tensor(torch.cov(cluster_samples.T), dtype=torch.float32)))
        if cluster_samples.size(0) < 2:
            theta_k['sigma'] = torch.eye(D) * 1e-6#*torch.rand(1)
        theta_k['v'] = torch.tensor(3.0)
        Theta.append(theta_k)

    m0 = X.mean()
    # (N-1)*
    S0 =K**(-1/D)*torch.diag(torch.diag(torch.cov(X.t())))
    #S0_hat = K**(-1/D)*torch.diag(torch.diag(torch.cov(X.t())))

    '''
    K = len(Theta)
    D = X.shape[1]
    sigma = calaculate_sigma(X, Theta)  # to cuda
    xi_i_k = calculate_xi(X, Theta)  # to cuda
    zeta_i_k = calculate_zeta(X, Theta, sigma)  # to cuda

    alpha_k = calculate_alpha_k(xi_i_k, alpha0_hat)  # to cuda
    Omega = torch.mul(xi_i_k, zeta_i_k)  # to cuda
    S_mle_k_list = []
    for k in range(K):
        p_ik = xi_i_k[:, k] 
        q_ik = zeta_i_k[:, k]    
        N_ik = p_ik*q_ik
        N_k = torch.sum(N_ik)
           
        x_bar_k = torch.mul(N_ik.unsqueeze(dim=1).repeat(1, D), X).sum(dim=0) / (N_k + 1e-6)
        
        kappa_k = kappa0_hat + N_k
        m_k = x_bar_k
        
        Omega_ik = Omega[:, k]              
        S_mle_k = calculate_S_mle_k(X, Omega_ik, x_bar_k)
        S_mle_k_list.append(S_mle_k)
    
    '''
    alpha0_hat = alpha0 * torch.ones(K)
    m0_hat = m0 
    kappa0_hat = kappa0
    #S0_hat = S0 + torch.sum(torch.stack(S_mle_k_list), dim=0)
    #S0_hat = torch.diag(torch.mean(torch.stack([torch.diag(theta['sigma'])  for theta in Theta]),dim=0))
    S0_hat = S0
    rho0_hat = 2*rho0

    return Theta, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat, cluster_labels



def update_SMM_parameters(X, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat,xi_i_k):
    K = len(Theta_prev)
    D = X.shape[1]
    sigma = calaculate_sigma(X, Theta_prev).to(device)  # to cuda
    #xi_i_k = calculate_xi(X, Theta_prev).to(device)  # to cuda
    zeta_i_k = calculate_zeta(X, Theta_prev, sigma).to(device)  # to cuda

    alpha_k = calculate_alpha_k(xi_i_k, alpha0_hat).to(device)  # to cuda
    Omega = torch.mul(xi_i_k, zeta_i_k).to(device)  # to cuda

    for k in range(K):
        p_ik = xi_i_k[:, k]
        q_ik = zeta_i_k[:, k]    
        N_ik = p_ik*q_ik
        N_k = torch.sum(N_ik)
           
        x_bar_k = torch.mul(N_ik.unsqueeze(dim=1).repeat(1, D), X).sum(dim=0) / (N_k+1e-6)

        
        kappa_k = kappa0_hat + N_k
        m_k = x_bar_k
        
        Omega_ik = Omega[:, k]              
        S_mle_k = calculate_S_mle_k(X, Omega_ik, x_bar_k)
        #print('S_mle_k', S_mle_k)

        S_k =S0_hat+ S_mle_k
        rho_k = rho0_hat + torch.sum(p_ik)
        if torch.sum(torch.isnan(p_ik)):
            print(p_ik)
        
        Theta_prev[k]['mu'] = m_k
        beta = rho0_hat/(rho_k)
        Theta_prev[k]['sigma'] = beta*S0_hat/rho0_hat+ (1-beta)*(torch.diag(torch.diag(S_mle_k)+1e-6)/(torch.sum(p_ik)+1e-6))
        Theta_prev[k]['v'] = caculate_v_k(Theta_prev[k]['v'], k, xi_i_k, zeta_i_k)
        Theta_prev[k]['pai'] = (alpha_k[k]-1+1e-6)/(alpha_k.sum()-K+1e-6)#*torch.rand(1).to('cuda')
        a = alpha_k.sum()
       
    return Theta_prev


