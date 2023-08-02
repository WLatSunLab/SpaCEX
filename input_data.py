import numpy as np
from keras.datasets import mnist
import torch
from torch.utils.data import Dataset


def load_mnist():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x = np.concatenate((x_train, x_test))
    y = np.concatenate((y_train, y_test)).astype(np.int32)
    x = x.astype(np.float32)
    x = np.divide(x, 255.)
    print('MNIST samples', x.shape)
    return x, y


class MnistDataset(Dataset):

    def __init__(self):
        self.x, self.y = load_mnist()

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, idx):
        return (torch.from_numpy(np.array(self.x[idx]))).unsqueeze(0), torch.from_numpy(
            np.array(self.y[idx])), torch.from_numpy(np.array(idx))


