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

'''CAE_encoder
For MNIST data sets
'''

'''Example
import CAE
from input_data import MnistDataset

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset()
model = CAE()
model.to(device)
model.pretrain(dataset)
data = dataset.x
y = dataset.y
data = torch.Tensor(data).to(device)
data=data.unsqueeze(1)
x_bar, hidden = model(data)
'''

def con3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

# 定义BasicBlock
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # 下面定义BasicBlock中的各个层
        self.conv1 = con3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        # inplace为True表示进行原地操作，一般默认为False，表示新建一个变量存储操作
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = con3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.dowansample = downsaple
        self.stride = stride

    # 定义前向传播函数将前面定义的各层连接起来
    def forward(self, x):
        identity = x  # 这是由于残差块需要保留原始输入

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.dowansample is not None:  # 这是为了保证原始输入与卷积后的输出层叠加时维度相同
            identity = self.dowansample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=10):
        # layers=参数列表 block选择不同的类
        self.inplanes = 64
        super(ResNet, self).__init__()
        # 1.conv1
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # 2.conv2_x
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        # 3.conv3_x
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        # 4.conv4_x
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        # 5.conv5_x
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AvgPool2d(7)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # 初始化权重
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        # 每个blocks的第一个residual结构保存在layers列表中。
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))
            # 该部分是将每个blocks的剩下residual 结构保存在layers列表中，这样就完成了一个blocks的构造。

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        #x = self.avgpool(x)
        x = x.view(x.size(0), -1)  # 将输出结果展成一行
        # x = self.fc(x)

        return x

def resnet18(pretrained=False):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2])
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model

def pretrain_cae(model, dataset):
    '''
    pretrain convolutional autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=0.001)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    for epoch in range(50):
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
        torch.save(model.state_dict(), 'cae_mnist.pkl')
    print("model saved to {}.".format('cae_mnist.pkl'))

class CAE_coder(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(CAE_coder, self).__init__()

        # encoder
        self.encoder = resnet18(pretrained=False)

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

        z = self.encoder(x)

        # decoder

        x_bar = self.decoder(z)

        return x_bar, z

class CAE(nn.Module):

    def __init__(self,
                 n_enc_1=500,
                 n_enc_2=500,
                 n_enc_3=1000,
                 n_dec_1=500,
                 n_dec_2=500,
                 n_dec_3=500,
                 n_input=784,
                 n_z=512,
                 #n_clusters=10,
                 alpha=1,
                 pretrain_path='cae_mnist.pkl'):
        super(CAE, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.cae = CAE_coder(
            n_enc_1=n_enc_1,
            n_enc_2=n_enc_2,
            n_enc_3=n_enc_3,
            n_dec_1=n_dec_1,
            n_dec_2=n_dec_2,
            n_dec_3=n_dec_3,
            n_input=n_input,
            n_z=n_z)
        # cluster layer
        #self.cluster_layer = Parameter(torch.Tensor(n_clusters, n_z))
        #torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def pretrain(self, dataset, path=''):
        if path == '':
            pretrain_cae(self.cae, dataset)
        # load pretrain weights
        self.cae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained cae from', self.pretrain_path)

    def forward(self, x):

        x_bar, hidden = self.cae(x)
        # cluster

        return x_bar, hidden


