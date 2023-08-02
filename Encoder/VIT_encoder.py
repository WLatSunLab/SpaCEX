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

'''VIT_encoder
For MNIST data sets
'''

'''Example
import VIT
from input_data import MnistDataset

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset()
model = VIT()
model.to(device)
model.pretrain(dataset)
data = dataset.x
y = dataset.y
data = torch.Tensor(data).to(device)
data=data.unsqueeze(1)
x_bar, hidden = model(data)
'''

class PatchEmbed(nn.Module):
    def __init__(self, img_size=(28,28), patch_size=(7,7), in_chans=1, embed_dim=49):
        super(PatchEmbed,self).__init__()
        # 原始大小为int，转为tuple，即：img_size原始输入224，变换后为[224,224]
        #img_size = to_2tuple(img_size)
        #patch_size = to_2tuple(patch_size)
        # 图像块的个数
        num_patches = (img_size[1] // patch_size[1]) * \
            (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
        # 输入维度为3，输出维度为块向量长度
        # 与原文中：分块、展平、全连接降维保持一致
        # 输出为[B, C, H, W]
        self.proj = nn.Conv2d(
            in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x).flatten(2).transpose(2, 1)
        return x

class MLP(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, input):
        output = self.net(input)
        return output

class MSA(nn.Module):
    """
    dim就是输入的维度，也就是embeding的宽度
    heads是有多少个patch
    dim_head是每个patch要多少dim
    dropout是nn.Dropout()的参数
    """
    def __init__(self, dim, heads=4, dim_head=7, dropout=0.,attn_drop=0.):
        super(MSA, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head=dim_head
        self.dropout = dropout

        # 论文里面的Dh
        self.Dh = dim_head ** -0.5

        # self-attention里面的Wq，Wk和Wv矩阵
        inner_dim = dim_head * heads
        self.inner_dim=inner_dim
        self.linear_q = nn.Linear(dim, inner_dim, bias=False)
        self.linear_k = nn.Linear(dim, inner_dim, bias=False)
        self.linear_v = nn.Linear(dim, inner_dim, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.output = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, input):
        """
        :param input: 输入是embeding，[batch, N, D]
        :return: MSA的计算结果的维度和输入维度是一样的
        """

        # 首先计算q k v
        # [batch, N, inner_dim]
        q = self.linear_q(input)
        k = self.linear_k(input)
        v = self.linear_v(input)
        # 转换为多头
        new_shape = q.size()[:-1] + (self.heads, self.dim_head)
        q=q.view(new_shape)
        k=k.view(new_shape)
        v=v.view(new_shape)
        q=torch.transpose(q,-3,-2)
        k=torch.transpose(k,-3,-2)
        v=torch.transpose(v,-3,-2) #[batch, head, N, head_size]
        # 接着计算矩阵A

        A = torch.matmul(q, torch.transpose(k,-2,-1)) * self.Dh
        A = torch.softmax(A, dim=-1) # [batch,head, N, N]
        A = self.attn_drop(A)
        SA = torch.matmul(A, v)#[batch,head, N, head_size]
        #多头拼接
        SA=torch.transpose(SA,-3,-2)#[batch, N,head, head_size]
        new_shape = SA.size()[:-2] + (self.inner_dim,)
        SA=SA.reshape(new_shape)# [batch, N, inner_dim]
        out = self.output(SA)# [batch, N, D]
        return out

class TransformerEncoder(nn.Module):
    def __init__(self, dim, hidden_dim):
        super(TransformerEncoder, self).__init__()
        self.norm = nn.LayerNorm(dim)
        self.msa = MSA(dim)
        self.mlp = MLP(dim, hidden_dim)
    def forward(self, input):
        output = self.norm(input)
        output = self.msa(output)
        output_s1 = output + input
        output = self.norm(output_s1)
        output = self.mlp(output)
        output_s2 = output + output_s1
        return output_s2

class VIT_emb(nn.Module):
    def __init__(self, num_patches,dim,  num_classes=10, num_layers=10):
        super(VIT_emb, self).__init__()
        hidden_dim=4*dim
        self.emb=PatchEmbed()
        # Position Embeddings
        self.pos_emb = nn.Parameter(torch.randn(1, num_patches, dim))
        self.layers = nn.ModuleList([])
        for _ in range(num_layers):
            self.layers.append(TransformerEncoder(dim, hidden_dim))
        self.pic2pic_attn=TransformerEncoder(dim, hidden_dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x):
        x=self.emb(x)

        x = x + self.pos_emb[:, :x.size(1), :]  # (1, num_patches, embed_dim)
        #x= nn.Dropout(0.9)(x)
        for layer in self.layers:
            x = layer(x)
        x = x.mean(dim=1)
        #x=self.pic2pic_attn(x)
        #x = self.mlp_head(x)
        return x



class VIT_coder(nn.Module):

    def __init__(self, n_enc_1, n_enc_2, n_enc_3, n_dec_1, n_dec_2, n_dec_3,
                 n_input, n_z):
        super(VIT_coder, self).__init__()

        # encoder
        self.encoder = VIT_emb(16,49)


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

def pretrain_vitae(model, dataset):
    '''
    pretrain vision transformer autoencoder
    '''
    train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
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
        torch.save(model.state_dict(), 'vit_mnist.pkl')
    print("model saved to {}.".format('vit_mnist.pkl'))

class VIT(nn.Module):

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
        super(VIT, self).__init__()
        self.alpha = 1.0
        self.pretrain_path = pretrain_path

        self.vitae = VIT_coder(n_enc_1=n_enc_1,n_enc_2=n_enc_2,n_enc_3=n_enc_3,n_dec_1=n_dec_1,n_dec_2=n_dec_2,n_dec_3=n_dec_3,n_input=n_input,n_z=n_z)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(10, n_z))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, dataset, path=''):
        if path == '':
            pretrain_vitae(self.vitae, dataset)
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

