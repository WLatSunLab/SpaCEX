import torch


'''
print a tsne dimensionality reduction image verifies whether the results of the embedding are meaningful
'''


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
    projections = projections.reshape(num,-1)
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
