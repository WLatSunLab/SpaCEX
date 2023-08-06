import torch.nn as nn
import math
import argparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils import model_zoo
from torch.utils.data import DataLoader
from torch.nn import Linear
from torchvision.models.video.resnet import model_urls

batchsize = 256
lr = 0.001
n_input = 784

'''VGG_encoder
For MNIST data sets
'''

'''Example
import VGG
from input_data import MnistDataset

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset()
model = VGG()
model.to(device)
model.pretrain(dataset)
data = dataset.x
y = dataset.y
data = torch.Tensor(data).to(device)
data=data.unsqueeze(1)
x_bar, hidden = model(data)
'''

class VGG_emb(nn.Module):  
    """
    Based on - https://github.com/kkweon/mnist-competition
    """
    def two_conv_pool(self, in_channels, f1, f2):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
    
    def three_conv_pool(self,in_channels, f1, f2, f3):
        s = nn.Sequential(
            nn.Conv2d(in_channels, f1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f1),
            nn.ReLU(inplace=True),
            nn.Conv2d(f1, f2, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f2),
            nn.ReLU(inplace=True),
            nn.Conv2d(f2, f3, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(f3),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        for m in s.children():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        return s
        
    
    def __init__(self, num_classes=10):
        super(VGG, self).__init__()
        self.l1 = self.two_conv_pool(1, 64, 64)
        self.l2 = self.two_conv_pool(64, 128, 128)
        self.l3 = self.three_conv_pool(128, 256, 256, 256)
        self.l4 = self.three_conv_pool(256, 256, 256, 512)
        
        self.classifier = nn.Sequential(
            nn.Dropout(p = 0.5),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(p = 0.5),
            nn.Linear(512, num_classes),
        )
    
    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x = x.view(x.size(0), -1)
        return x 

class VGG_coder(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(VGG_coder, self).__init__()

        # encoder
        self.encoder = VGG_emb()
        
        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(n_z, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
            nn.Unflatten(dim=1,unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 3,stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2,padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2,padding=1, output_padding=1)
)


    def forward(self, x):

        # encoder
       
        hidden = self.encoder(x)

        # decoder
        
        x_bar = self.decoder(hidden)

        return x_bar, hidden


def pretrain_vggae(model):
    '''
    pretrain convolutional autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.001)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    for epoch in range(100):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, z = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), 'vgg_mnist.pkl')
    print("model saved to {}.".format('vgg_mnist.pkl'))


class VGG(nn.Module):

    def __init__(self,
                 n_enc_1=500,
                 n_enc_2=500,
                 n_enc_3=1000,
                 n_dec_1=1000,
                 n_dec_2=500,
                 n_dec_3=500,
                 n_input=784,
                 n_z=49,
                 #n_clusters,
                 alpha=1,
                 pretrain_path='vitae_mnist.pkl'):
        super(VGG, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.vitae = VGG_coder(n_enc_1=n_enc_1,n_enc_2=n_enc_2,n_enc_3=n_enc_3,n_dec_1=n_dec_1,n_dec_2=n_dec_2,n_dec_3=n_dec_3,n_input=n_input,n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(10, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, dataset, path=''):
        if path == '':
            pretrain_vggae(self.vitae, dataset)
        # load pretrain weights
        self.vitae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained cae from', self.pretrain_path)

    def forward(self, x):

        x_bar, hidden = self.vitae(x)
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(hidden.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, hidden, q
