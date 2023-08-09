import torch
import numpy as np
import torch.nn as nn
import argparse
from sklearn.cluster import KMeans
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.nn import Linear


class PatchEmbed(nn.Module):  # 在MNIST上，[B, 1, 28, 28]->[B, 49, embed_dim]
    def __init__(self, img_size=(28, 28), patch_size=(4, 4), in_chans=1, embed_dim=16):
        super(PatchEmbed, self).__init__()

        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])  # 获取49个patch

        # kernel_size=块大小，即每个块输出一个值，类似每个块展平后使用相同的全连接层进行处理
        # 输入维度为3，输出维度为块向量长度
        # 与原文中：分块、展平、全连接降维保持一致
        # 输出为[B, C, H, W]
        # 在MNIST上输出位[B, embed_dim, 7, 7]
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            "Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # [B, C, H, W] -> [B, C, H*W] ->[B, H*W, C]
        x = self.proj(x).flatten(2).transpose(2, 1)  # 在MNIST上为[B, 49, embed_dim]
        return x


class MLP(nn.Module):  # 简单的autoencoder，特征不变
    def __init__(self, dim, hidden_dim, dropout=0.):
        super(MLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        output = self.net(x)
        return output


class MSA(nn.Module):
    """
    dim就是输入的维度，也就是embeding的宽度
    heads是有多少个patch
    dim_head是每个patch要多少dim
    dropout是nn.Dropout()的参数
    """

    def __init__(self, dim, heads=4, dim_head=2, dropout=0., attn_drop=0., qkv_bias=False):
        super(MSA, self).__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.dropout = dropout

        # 论文里面的Dh
        self.Dh = dim_head ** -0.5

        # self-attention里面的Wq，Wk和Wv矩阵
        inner_dim = dim_head * heads
        self.inner_dim = inner_dim
        self.linear_q = nn.Linear(dim, inner_dim, bias=qkv_bias)
        self.linear_k = nn.Linear(dim, inner_dim, bias=qkv_bias)
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
        q = q.view(new_shape)
        k = k.view(new_shape)
        v = v.view(new_shape)
        q = torch.transpose(q, -3, -2)
        k = torch.transpose(k, -3, -2)
        v = torch.transpose(v, -3, -2)  # [batch, head, N, head_size]
        # 接着计算矩阵A

        A = torch.matmul(q, torch.transpose(k, -2, -1)) * self.Dh
        A = torch.softmax(A, dim=-1)  # [batch,head, N, N]
        A = self.attn_drop(A)
        SA = torch.matmul(A, v)  # [batch,head, N, head_size]
        # 多头拼接
        SA = torch.transpose(SA, -3, -2)  # [batch, N,head, head_size]
        new_shape = SA.size()[:-2] + (self.inner_dim,)
        SA = SA.reshape(new_shape)  # [batch, N, inner_dim]
        out = self.output(SA)  # [batch, N, D]
        return out


class Block(nn.Module):
    def __init__(self, dim, num_heads, dim_head, mlp_ratio, qkv_bias=True):
        super(Block, self).__init__()
        hidden_dim = mlp_ratio * dim
        self.norm = nn.LayerNorm(dim)
        self.msa = MSA(dim, heads=num_heads, dim_head=dim_head, qkv_bias=qkv_bias)
        self.mlp = MLP(dim, hidden_dim)

    def forward(self, input):
        output = self.norm(input)
        output = self.msa(output)
        output_s1 = output + input
        output = self.norm(output_s1)
        output = self.mlp(output)
        output_s2 = output + output_s1
        return output_s2

# get positional embedding
def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    return pos_embed

def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb

def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


class MAE_encoder(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size=(28, 28),
                 patch_size=(4, 4),
                 in_chans=1,
                 embed_dim=16,
                 depth=3,
                 num_heads=4,
                 dim_head=4,
                 decoder_embed_dim=16,
                 mlp_ratio=4.,
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # MAE encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)  # Bx1x28x28->Bx49xemben_dim
        num_patches = self.patch_embed.num_patches
        # self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim),
                                      requires_grad=True)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim, num_heads, dim_head, mlp_ratio, qkv_bias=True)
            for _ in range(depth)])

        # --------------------------------------------------------------------------
        # MAE decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pos_embed = nn.Parameter(torch.zeros(1, num_patches, decoder_embed_dim),
                                              requires_grad=True)  # fixed sin-cos embedding

        # decoder
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 3 * 3 * 32),
            nn.ReLU(True),
            nn.Unflatten(dim=1, unflattened_size=(32, 3, 3)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, output_padding=0),
            nn.BatchNorm2d(16),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 3, stride=2, padding=1, output_padding=1)
        )

        # --------------------------------------------------------------------------

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.patch_embed.num_patches ** .5),
                                            cls_token=False)
        # self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        decoder_pos_embed = get_2d_sincos_pos_embed(self.decoder_pos_embed.shape[-1],
                                                    int(self.patch_embed.num_patches ** .5), cls_token=False)
        # self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        # torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2 *3)
        """
        p = self.patch_embed.patch_size[0]  # 4
        h = imgs.shape[2] // p  # 7
        w = imgs.shape[3] // p  # 7
        imgs = imgs[:, :, :h * p, :w * p]  # [:, :, :28, :28]
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p ** 2 * 1))  # 在这里emb_dim=16
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        p = self.patch_embed.patch_size[0]
        h = w = int(x.shape[1] ** .5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, 3))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], 1, h * p, h * p))
        return imgs

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def forward_encoder(self, x, mask_ratio):

        x = self.patch_embed(x)  # Bx1x28x28->Bx49xpatch_size**2x1
        x = x + self.pos_embed[:, :, :]  # 获取位置编码
        x1 = x.clone()
        x, mask, ids_restore = self.random_masking(x, mask_ratio)  # 获取mask结果

        for blk in self.blocks:
            x = blk(x)
        for blk in self.blocks:
            x1 = blk(x1)
        return x1, x, mask, ids_restore  # 两次Transformer， 一次Transformer， mask编码

    def forward_decoder(self, x, ids_restore):
        # embed tokens
        # x = self.decoder_embed(x)

        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(x.shape[0], ids_restore.shape[1] - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x.shape[2]))  # unshuffle
        # add pos embed
        x = x + self.decoder_pos_embed
        x = x.mean(dim=1)
        x = self.decoder(x)

        return x.reshape((x.shape[0], ids_restore.shape[1], -1))

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p*1]
        mask: [N, L], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6) ** .5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        whole_latent, latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        z = whole_latent.mean(dim=1)
        return z, loss, pred, mask

class MAE(nn.Module):

    def __init__(self,
                 img_size=(28, 28),
                 patch_size=(4, 4),
                 in_chans=1,
                 embed_dim=16,
                 depth=3,
                 num_heads=4,
                 dim_head=4,
                 decoder_embed_dim=16,
                 mlp_ratio=4.,
                 norm_pix_loss=False,
                 alpha=1,
                 n_clusters=10,
                 pretrain_path='mae_mnist.pkl'):
        super(MAE, self).__init__()
        self.alpha = alpha
        self.pretrain_path = pretrain_path
        self.mae =MAE_encoder(
                img_size=img_size,
                patch_size=patch_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                dim_head=dim_head,
                decoder_embed_dim=decoder_embed_dim,
                mlp_ratio=mlp_ratio,
                norm_pix_loss=norm_pix_loss)
        # cluster layer
        self.cluster_layer = Parameter(torch.Tensor(n_clusters, patch_size[0]**2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

    def pretrain(self, dataset, batch_size, lr, pretrain=True):
        if pretrain:
            pretrain_mae(self.mae, dataset, batch_size, lr)
        # load pretrain weights
        self.mae.load_state_dict(torch.load(self.pretrain_path))
        print('load pretrained mae from', self.pretrain_path)

    def forward(self, x):

        z, loss, x_bar, mask = self.mae(x)
        # cluster
        q = 1.0 / (1.0 + torch.sum(
            torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
        q = q.pow((self.alpha + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return x_bar, q, loss


def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()


def pretrain_mae(model, dataset, batch_size, lr):
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(model)
    cuda = torch.cuda.is_available()
    device = torch.device("cuda" if cuda else "cpu")
    optimizer = Adam(model.parameters(), lr=lr)
    for epoch in range(100):
        total_loss = 0.
        for batch_idx, (x, _, _) in enumerate(train_loader):
            x = x.to(device)

            optimizer.zero_grad()
            z,loss,x_bar,_  = model(x)
            #loss = F.mse_loss(x_bar, x)
            total_loss += loss.item()

            loss.backward()
            optimizer.step()

        print("epoch {} loss={:.4f}".format(epoch,
                                            total_loss / (batch_idx + 1)))
        torch.save(model.state_dict(), 'mae_mnist.pkl')
    print("model saved to {}.".format('mae_mnist.pkl'))


