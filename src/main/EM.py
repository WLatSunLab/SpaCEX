import torch
from scipy.stats import mvn
from SMM import update_SMM_parameters
from SMM import calculate_xi
from SMM import calculate_mu


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
    N = len(X)
    xi_i_k = calculate_xi(X, Theta_prev)
    xi_i_k_history = [xi_i_k] 

    for i in range(max_iterations):
        #p_ik = calculate_xi(X, Theta_prev)

        # Update mu
        #mu_k = calculate_mu(X, p_ik)

        # Update sigma
        #sigma_k = calculate_sigma(X, p_ik, mu_k)

        # Update v
        #v_k = calculate_v(X, p_ik)

        # Update pai
        # pai_k = calculate_pai(xi_i_k, p_ik)

        # Update Theta
        #for k in range(len(Theta_prev)):
            #Theta_prev[k]['mu'] = mu_k[k]

            #Theta_prev[k]['sigma'] = torch.abs(torch.diag(torch.diag(sigma_k[k]) + 1e-6))
            #Theta_prev[k]['sigma'] = torch.where(torch.isnan(Theta_prev[k]['sigma']), torch.eye(X.size(1)),
                                                 #Theta_prev[k]['sigma'])

            #Theta_prev[k]['v'] = v_k[k]
            # Theta_prev[k]['pai'] = pai_k[k]

        Theta_prev = update_SMM_parameters(X, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat)  # 更新参数
        # for the in Theta_prev:
        # print(the['pai'])
        clusters_new = assign_clusters(X, Theta_prev)

        # Tolerance discrimination
        if i > 0 and torch.sum((clusters_new - clusters) != 0) / N < tol:
            break

        clusters = clusters_new

        xi_i_k_history.append(xi_i_k)  

    if max_iterations > 0:
        s = torch.unique(clusters)
        ss = []
        for i in s:
            if torch.sum(clusters == i) > N // 10:
                ss.append(i)
        ss = torch.tensor(ss)
        if len(s) < config['num_classes']:
            for i in range(config['num_classes']):
                if i not in s:
                    j = ss[torch.randint(high=len(ss), size=(1,))]
                    Theta_prev[i] = Theta_prev[j]

            print('Reassign {}'.format(config['num_classes'] - len(s)), 'clusters')

    return Theta_prev, clusters, xi_i_k_history



