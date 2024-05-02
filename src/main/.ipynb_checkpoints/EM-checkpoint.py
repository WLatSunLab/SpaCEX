import torch
import time
from scipy.stats import mvn
from SpaCEX.src.main.SMM import update_SMM_parameters
from SpaCEX.src.main.SMM import calculate_xi


def assign_clusters(X, Theta_updated):
    xi_i_k = calculate_xi(X, Theta_updated)
    clusters = torch.argmax(xi_i_k, dim=1)
    return clusters


def EM_algorithm(X, K, 
                 Theta_prev,
                 alpha0_hat,
                 m0_hat,
                 kappa0_hat,
                 S0_hat,
                 rho0_hat,
                 clusters,
                 max_iterations,
                 config,
                 tol=5 * 1e-3):
    
    X = X.to('cuda:1')
    alpha0_hat = torch.tensor(alpha0_hat).to('cuda:1')
    m0_hat = torch.tensor(m0_hat).to('cuda:1')
    kappa0_hat = torch.tensor(kappa0_hat).to('cuda:1')
    S0_hat = torch.tensor(S0_hat).to('cuda:1')
    rho0_hat = torch.tensor(rho0_hat).to('cuda:1')
    clusters = torch.tensor(clusters).to('cuda:1')
    
    N = len(X)
    xi_i_k = calculate_xi(X, Theta_prev,clusters)
    xi_i_k_history = [xi_i_k] 
    a = time.time()
    for i in range(max_iterations):
        #print('iterations', i)
        Theta_prev = update_SMM_parameters(X, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat,xi_i_k)  # 更新参数
        xi_i_k = calculate_xi(X, Theta_prev,clusters)
        clusters_new = torch.argmax(xi_i_k, dim=1)
        # Tolerance discrimination
        if i > 0 and torch.sum((clusters_new - clusters) != 0) / N < tol:
            break

        clusters = clusters_new

        xi_i_k_history.append(xi_i_k)  
    b = time.time()
    if max_iterations > 0:
        s = torch.unique(clusters)
        ss = []
        for i in s:
            if torch.sum(clusters == i) > N // 50:
                ss.append(i)
        ss = torch.tensor(ss)
        if len(s) < config['num_classes'] and len(ss)!=0:
            for i in range(config['num_classes']):
                if i not in s:
                    j = ss[torch.randint(high=len(ss), size=(1,))]
                    Theta_prev[i]['mu'] = 0.99*Theta_prev[j]['mu']
                    Theta_prev[i]['pai'] = 0.99*Theta_prev[j]['pai']
                    Theta_prev[i]['sigma'] = 0.981*Theta_prev[j]['sigma']
                    Theta_prev[i]['v'] = 0.99*Theta_prev[j]['v']

            #print('Reassign {}'.format(config['num_classes'] - len(s)), 'clusters')
        xi_i_k = calculate_xi(X, Theta_prev,clusters)
        xi_i_k_history.append(xi_i_k) 
        clusters = torch.argmax(xi_i_k, dim=1)

    return Theta_prev, clusters, xi_i_k_history



