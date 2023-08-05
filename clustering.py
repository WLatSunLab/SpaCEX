import torch
import numpy as np
import torch.nn.functional as F
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment as linear_assignment
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from torch.optim import Adam
from torch.utils.data import DataLoader
from _config import Config


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


def clustering(model, dataset, config):
    acc_train=[]
    nmi_train=[]
    ari_train=[]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    train_loader = DataLoader(
        dataset, batch_size=config['batch_size'], shuffle=False)
    optimizer = Adam(model.parameters(), lr=config['lr'])

    # cluster parameter initiate
    data = dataset.x
    y = dataset.y
    data = torch.Tensor(data).to(device)
    data = data.unsqueeze(1)
    x_bar, hidden, _ = model.cae(data)

    kmeans = KMeans(n_clusters=10, n_init=config['n_init'])
    y_pred = kmeans.fit_predict(hidden.data.cpu().numpy())
    nmi_k = nmi_score(y_pred, y)
    print("nmi score={:.4f}".format(nmi_k))

    hidden = None
    x_bar = None

    y_pred_last = y_pred
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)

    model.train()
    for epoch in range(config['n_epochs']):

        if epoch % config['interval'] == 0:

            _, _, tmp_q = model(data)

            # update target distribution p
            tmp_q = tmp_q.data
            p = target_distribution(tmp_q)

            # evaluate clustering performance
            y_pred = tmp_q.cpu().numpy().argmax(1)
            delta_label = np.sum(y_pred != y_pred_last).astype(
                np.float32) / y_pred.shape[0]
            y_pred_last = y_pred

            acc = cluster_acc(y, y_pred)
            nmi = nmi_score(y, y_pred)
            ari = ari_score(y, y_pred)
            acc_train.append(acc)
            nmi_train.append(acc)
            ari_train.append(acc)
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

            x_bar, hidden, q = model(x)

            reconstr_loss = F.mse_loss(x_bar, x)
            kl_loss = F.kl_div(q.log(), p[idx])
            loss = 0.1 * kl_loss + reconstr_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return y_pred_last, acc_train, nmi_train, ari_train

# Projections in 2D after Contrastive Learning, 25 epochs
def tsne_print(model, dataset, num, path):
    model.eval()
    projections = []
    labels=[]
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")

    # Loop through the train data loader
    for images, l, _ in dataset:
        # Move data to device
        images = images.unsqueeze(1)

        images = images.to(device)
        #images = train_transform(images)

        l=l.to(device)

        # Compute embeddings
        with torch.no_grad():
            x_bar, hidden,_ = model(images)

        # Append embeddings to lists
        projections.append(hidden.cpu().numpy())
        labels.append(l.cpu().numpy())

    # Concatenate embeddings from all batches
    import numpy as np
    projections = np.array(projections)
    projections = projections.reshape(10000,-1)
    labels = np.array(labels)

    print(projections.shape, labels.shape)

    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    from matplotlib.colors import ListedColormap
    import numpy as np

    # Reduce dimensionality of features using t-SNE
    tsne = TSNE(n_components=2, verbose=1)
    features_tsne = tsne.fit_transform(projections)
    color_map = ListedColormap(['tab:blue', 'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink',                                       'tab:gray', 'tab:olive', 'tab:cyan'])
    fig = plt.figure(figsize=(12,12))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(features_tsne[:,0], features_tsne[:,1], c=labels, cmap=color_map)
    ax.set_xlabel('Dimension 1')
    ax.set_ylabel('Dimension 2')

    # Add a color bar to the plot to show the label-color mapping
    cbar = plt.colorbar(scatter, ticks=np.unique(labels))
    cbar.ax.set_yticklabels(np.unique(labels))

    ax.legend()
    plt.savefig(path+'.png', dpi=300)
    plt.show()



