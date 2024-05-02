import torch
# likelihood
def likelihood(e, ceta):
    e = torch.log(e @ ceta).sum() / e.size(0)

    return -e


# Lsep
def sep(mean, k):
    s = torch.tensor(0.0)

    for i in range(mean.size(0)):
        sorted, _ = torch.sort(torch.pow(mean - mean[i], 2).sum(1))
        s += sorted[1:1 + k].sum()
    return s


# re
def regularization(h, lap_mat):
    return (h.T @ lap_mat @ h).trace()


# size
def size(q):
    n = q.size()[0]
    n_clust = q.size()[1]
    p1 = torch.zeros((n_clust,)).to('cuda:1')
    p1 = torch.sum(q, dim=0) / n
    #for i in range(n_clust):
        #if p1[i] > 1 / 3000:
            #p1[i] = 1

    return -(torch.sum((-p1) * torch.log(p1)))


def R(sigma):
    s = torch.tensor(0.0)
    for i in range(sigma.size(0)):
        s += torch.sum(1 / torch.diag(sigma[i]))
    return s
