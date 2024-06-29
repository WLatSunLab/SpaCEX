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
    
    X = X.to('cuda')
    alpha0_hat = torch.tensor(alpha0_hat).to('cuda')
    m0_hat = torch.tensor(m0_hat).to('cuda')
    kappa0_hat = torch.tensor(kappa0_hat).to('cuda')
    S0_hat = torch.tensor(S0_hat).to('cuda')
    rho0_hat = torch.tensor(rho0_hat).to('cuda')
    clusters = torch.tensor(clusters).to('cuda')
    
    N = len(X)
    xi_i_k = calculate_xi(X, Theta_prev)
    xi_i_k_history = [xi_i_k] 
    a = time.time()
    for i in range(max_iterations):
        print('iterations', i)
        Theta_prev = update_SMM_parameters(X, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat)  # 更新参数
        clusters_new = assign_clusters(X, Theta_prev)

        # Tolerance discrimination
        if i > 0 and torch.sum((clusters_new - clusters) != 0) / N < tol:
            break

        clusters = clusters_new
        counts = torch.bincount(clusters)
        #print(counts)
        xi_i_k_history.append(xi_i_k)  
    b = time.time()
    print(b-a, 'five00', (b-a)*50)
    if max_iterations > 0:
        s = torch.unique(clusters)
        ss = []
        for i in s:
            if torch.sum(clusters == i) > N // 5:
                ss.append(i)
        ss = torch.tensor(ss)
        if len(s) < config['num_classes'] and len(ss)!=0:
            for i in range(config['num_classes']):
                if i not in s:
                    j = ss[torch.randint(high=len(ss), size=(1,))]
                    Theta_prev[i] = Theta_prev[j]

            #print('Reassign {}'.format(config['num_classes'] - len(s)), 'clusters')
    
    

    return Theta_prev, clusters, xi_i_k_history



