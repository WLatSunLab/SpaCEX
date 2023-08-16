import torch.nn as nn
import math
import argparse
import numpy as np
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


'''CAE_encoder
For MNIST data sets
'''


def con3x3(in_planes, out_planes, stride=1):  
    "3x3 convolution with padding"
    return nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=3,
                     stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsaple=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, basic_num=2):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")

        # Identify BasicBlock
        self.basic_num = basic_num
        self.conv1 = con3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
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

        if self.basic_num == 2:
            out = self.conv2(out)
            out = self.bn2(out)

        if self.dowansample is not None:  
            identity = self.dowansample(x)

        out = out + identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, block, layers,
                 in_channels=1,
                 conv1_outplanes=32,
                 bolck1_outplanes=64,
                 bolck2_outplanes=128,
                 bolck3_outplanes=256,
                 bolck4_outplanes=512,
                 basic_num=2,
                 layers_num=3,  
                 maxpool_dr=1,  
                 pool_bool=0,  
                 ):

        self.in_channels = in_channels
        self.inplanes = conv1_outplanes
        self.bolck1_outplanes = bolck1_outplanes
        self.bolck2_outplanes = bolck2_outplanes
        self.bolck3_outplanes = bolck3_outplanes
        self.bolck4_outplanes = bolck4_outplanes
        self.layers_num = layers_num
        self.maxpool_dr = maxpool_dr
        self.pool_bool = pool_bool
        super(ResNet, self).__init__()
        # 1.conv1, 1x28x28->conv1_outplanesx14x14
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=conv1_outplanes, kernel_size=3, stride=2,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(conv1_outplanes)
        self.relu = nn.ReLU(inplace=True)
        # 2.conv2_x, conv1_outplanesx14x14->bolck1_outplanesx7x7
        self.maxpool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1_1 = self._make_layer(block, planes=bolck1_outplanes, blocks=layers[0], stride=1,
                                         basic_num=basic_num)
        self.layer1_2 = self._make_layer(block, planes=bolck1_outplanes, blocks=layers[0], stride=2,
                                         basic_num=basic_num)
        # 3.conv3_x, bolck1_outplanesx7x7->bolck2_outplanesx4x4
        self.layer2 = self._make_layer(block, planes=bolck2_outplanes, blocks=layers[1], stride=2, basic_num=basic_num)
        # 4.conv4_x, bolck2_outplanesx4x4->bolck3_outplanesx2x2
        self.layer3 = self._make_layer(block, planes=bolck3_outplanes, blocks=layers[2], stride=2, basic_num=basic_num)
        # 5.conv5_x
        self.layer4 = self._make_layer(block, planes=bolck4_outplanes, blocks=layers[3], stride=2, basic_num=basic_num)
        self.avgpool1 = nn.AvgPool2d(7)
        self.avgpool2 = nn.AvgPool2d(4)
        self.avgpool3 = nn.AvgPool2d(2)
        self.maxpool2_1 = nn.MaxPool2d(7)
        self.maxpool2_2 = nn.MaxPool2d(4)
        self.maxpool2_3 = nn.MaxPool2d(2)

        # Initial weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1, basic_num=2):
        downsaple = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsaple = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes=planes, stride=stride,
                            downsaple=downsaple, basic_num=basic_num))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes=planes, basic_num=basic_num))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)  # 28x28->14x14
        x = self.bn1(x)
        x = self.relu(x)
        if self.maxpool_dr:
            x = self.maxpool1(x)
            x = self.layer1_1(x)
        else:
            x = self.layer1_2(x)
        if self.layers_num >= 1:
            if self.pool_bool == 0:
                n_z = self.bolck1_outplanes * 7 * 7
            elif self.pool_bool == 1: 
                x = self.avgpool1(x)
                n_z = self.bolck1_outplanes
            elif self.pool_bool == 2:  
                x = self.maxpool2_1(x)
                n_z = self.bolck1_outplanes
            if self.layers_num == 1:
                x = x.view(x.size(0), -1)
                return x, n_z
            elif self.layers_num >= 2:
                x = self.layer2(x)
                if self.pool_bool == 0:
                    n_z = self.bolck2_outplanes * 4 * 4
                elif self.pool_bool == 1:  
                    x = self.avgpool2(x)
                    n_z = self.bolck2_outplanes
                elif self.pool_bool == 2: 
                    x = self.maxpool2_2(x)
                    n_z = self.bolck2_outplanes
                if self.layers_num == 2:
                    x = x.view(x.size(0), -1)
                    return x, n_z
                elif self.layers_num >= 3:
                    x = self.layer3(x)
                    if self.pool_bool == 0:
                        n_z = self.bolck3_outplanes * 2 * 2
                    elif self.pool_bool == 1: 
                        x = self.avgpool3(x)
                        n_z = self.bolck3_outplanes
                    elif self.pool_bool == 2: 
                        x = self.maxpool2_3(x)
                        n_z = self.bolck3_outplanes
                    if self.layers_num == 3:
                        x = x.view(x.size(0), -1)
                        return x, n_z
                    elif self.layers_num == 4:
                        x = self.layer4(x)
                        if self.pool_bool == 0:
                            n_z = self.bolck4_outplanes * 1 * 1
                            x = x.view(x.size(0), -1)
                            return x, n_z 


def resnet18(in_channels=1,
             pretrained=False,
             basic_num=2,
             conv1_outplanes=32,
             bolck1_outplanes=64,
             bolck2_outplanes=128,
             bolck3_outplanes=256,
             bolck4_outplanes=512,
             layers_num=2,
             maxpool_dr=1,
             pool_bool=1):
    """Constructs a ResNet-18 model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(in_channels=in_channels,
                   block=BasicBlock,
                   layers=[2, 2, 2, 2],
                   basic_num=basic_num,
                   conv1_outplanes=conv1_outplanes,
                   bolck1_outplanes=bolck1_outplanes,
                   bolck2_outplanes=bolck2_outplanes,
                   bolck3_outplanes=bolck3_outplanes,
                   bolck4_outplanes=bolck4_outplanes,
                   layers_num=layers_num,
                   maxpool_dr=maxpool_dr,
                   pool_bool=pool_bool)
    if pretrained:
        model.load_state_dict(model_zoo.load_url(model_urls['resnet18']))
    return model


def pretrain_cae(model, dataset, batch_size, lr):
    '''
    pretrain convolutional autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(model)
    optimizer = Adam(model.parameters(), lr=lr)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    # model.to(device)
    for epoch in range(100):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            x_bar, hidden, _ = model(x)
            loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            # loss1 = loss.detach_().requires_grad_(True)
            # loss1.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), 'cae_mnist.pkl')
    print("model saved to {}.".format('cae_mnist.pkl'))


class CAE_coder(nn.Module):

    def __init__(self,
                 in_channels,
                 basic_num,
                 conv1_outplanes,
                 bolck1_outplanes,
                 bolck2_outplanes,
                 bolck3_outplanes,
                 bolck4_outplanes,
                 layers_num,
                 maxpool_dr,
                 pool_bool,
                 n_z
                 ):
        super(CAE_coder, self).__init__()

        # encoder, get Nxbolck3_outplanesx1
        self.encoder = resnet18(pretrained=False,
                                in_channels=in_channels,
                                basic_num=basic_num,
                                conv1_outplanes=conv1_outplanes,
                                bolck1_outplanes=bolck1_outplanes,
                                bolck2_outplanes=bolck2_outplanes,
                                bolck3_outplanes=bolck3_outplanes,
                                bolck4_outplanes=bolck4_outplanes,
                                layers_num=layers_num,
                                maxpool_dr=maxpool_dr,
                                pool_bool=pool_bool)

        # decoder
        if in_channels == 1:
            self.decoder = nn.Sequential(
                nn.Linear(in_features=n_z, out_features=128),
                nn.ReLU(True),
                nn.Linear(128, 3 * 3 * 32),
                nn.ReLU(True),
                # 1x3*3*32->32x3x3
                nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
                # 32x3x3->16x7x7
                nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                # 16x7x7->8x14x14
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                # 8x14x14->1x28x28
                nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
            )
        if in_channels == 3:
            self.decoder = nn.Sequential(
                nn.Linear(in_features=n_z, out_features=256),
                nn.ReLU(True),
                nn.Linear(256, 3 * 3 * 32),
                nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
                # 32x3x3->16x8x8
                nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=1),
                nn.BatchNorm2d(16),
                nn.ReLU(True),
                # 16x8x8->8x16x16
                nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(8),
                nn.ReLU(True),
                # 8x16x16->1x32x32
                nn.ConvTranspose2d(8, 3, 3, stride=2, padding=1, output_padding=1)
            )

    def forward(self, x):
      # encoder
      z, n_z = self.encoder(x)
      # decoder
      x_bar = self.decoder(z)

      return x_bar, z, n_z

class CAE(nn.Module):

    def __init__(self,
                in_channels=1,
                basic_num=2,
                conv1_outplanes=32,
                bolck1_outplanes=64,
                bolck2_outplanes=128,
                bolck3_outplanes=256,
                bolck4_outplanes=512,
                layers_num=3,
                maxpool_dr=1, 
                pool_bool=0,
                alpha=1.0,
                n_z=1024,
                pretrain_path='cae_mnist.pkl'
                ):
        super(CAE, self).__init__()
        self.pretrain_path = pretrain_path
        self.cae = CAE_coder(
            in_channels=in_channels,
            basic_num=basic_num,
            conv1_outplanes=conv1_outplanes,
            bolck1_outplanes=bolck1_outplanes,
            bolck2_outplanes=bolck2_outplanes,
            bolck3_outplanes=bolck3_outplanes,
            bolck4_outplanes=bolck4_outplanes,
            layers_num=layers_num,
            maxpool_dr=maxpool_dr, 
            pool_bool=pool_bool,
            n_z=n_z)
        # cluster layer
        self.alpha = alpha
        self.cluster_layer = Parameter(torch.Tensor(10, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, dataset, batch_size, lr, path=''):
        if path == '':
             pretrain_cae(self.cae, dataset, batch_size, lr)
        # load pretrain weights
        self.cae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained cae from', self.pretrain_path)

    def forward(self, x):
        x_bar, hidden, n_z = self.cae(x)
        # cluster_layer = Parameter(torch.Tensor(10, n_z))
        # torch.nn.init.xavier_normal_(cluster_layer.data)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
                torch.pow(hidden.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()

        return x_bar, hidden, q


