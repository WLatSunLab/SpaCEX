import torch
from scipy.stats import mvn


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


def calculate_alpha_k(xi_i_k, alpha0_hat):
    alpha_k = alpha0_hat + torch.sum(xi_i_k, dim=0)
    return alpha_k


def assign_clusters(X, Theta_updated):
    xi_i_k = calculate_xi(X, Theta_updated)
    clusters = torch.argmax(xi_i_k, dim=1)
    return clusters


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


def EM_algorithm(X, K, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat, clusters, max_iterations, config,
                 tol=5 * 1e-3):
    N = len(X)
    xi_i_k = calculate_xi(X, Theta_prev)
    xi_i_k_history = [xi_i_k]  # 用于保存每次迭代后的后验概率

    for i in range(max_iterations):
        p_ik = calculate_xi(X, Theta_prev)

        # Update mu
        mu_k = calculate_mu(X, p_ik)

        # Update sigma
        sigma_k = calculate_sigma(X, p_ik, mu_k)

        # Update v
        v_k = calculate_v(X, p_ik)

        # Update pai
        # pai_k = calculate_pai(xi_i_k, p_ik)

        # Update Theta
        for k in range(len(Theta_prev)):
            Theta_prev[k]['mu'] = mu_k[k]

            Theta_prev[k]['sigma'] = torch.abs(torch.diag(torch.diag(sigma_k[k]) + 1e-6))
            Theta_prev[k]['sigma'] = torch.where(torch.isnan(Theta_prev[k]['sigma']), torch.eye(X.size(1)),
                                                 Theta_prev[k]['sigma'])

            Theta_prev[k]['v'] = v_k[k]
            # Theta_prev[k]['pai'] = pai_k[k]

        Theta_prev = update_SMM_parameters(X, Theta_prev, alpha0_hat, m0_hat, kappa0_hat, S0_hat, rho0_hat)  # 更新参数
        # for the in Theta_prev:
        # print(the['pai'])
        clusters_new = assign_clusters(X, Theta_prev)

        # 如果相对变化小于阈值，则停止迭代
        if i > 0 and torch.sum((clusters_new - clusters) != 0) / N < tol:
            break

        clusters = clusters_new

        xi_i_k_history.append(xi_i_k)  # 保存每次迭代的后验概率

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



