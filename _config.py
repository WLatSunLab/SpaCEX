

class Config:
    def __init__(self,  dataset='MNIST', model='MAE'):
        config = {
            'MNIST': {
                'CAE': {
                    'model': 'CAE',  # 做模型标记，后续用作聚类判断
                    'in_channels': 1,  # 在MNIST上Channel为1
                    'batch_size': 256,  # 预训练时的batch_size
                    'lr': 0.001,  # 预训练时的学习率
                    'alpha': 1.0,  # DEC时的alpha
                    'n_epochs': 100,  # 预训练的epoch数
                    'tol': 0.001,  # DEC时优化容忍度
                    'num_classes': 10,  # 在MNIST上
                    'basic_num': 1,  # 决定basicblock上的卷积次数，至少为1
                    'conv1_outplanes': 32,  # 初始卷积参数
                    'bolck1_outplanes': 64,
                    'bolck2_outplanes': 128,
                    'bolck3_outplanes': 256,
                    'bolck4_outplanes': 512,
                    'layers_num': 2,  # layers的层数，至少为1
                    'maxpool_dr': 1,  # 0表示不用maxpooling降维
                    'pool_bool': 0,  # 0表示不pooling，1表示avg，2表示max
                    'n_init': 20,  # kmeans的初始参数
                    'interval': 1  # DEC时的训练间隔
                },
                'VIT': {},
                'MAE': {
                    'model': 'MAE',  # 做模型标记，后续用作聚类判断
                    'batch_size': 256,  # dddd
                    'lr': 0.001,  # dddd
                    'n_epochs': 100,  # dddd
                    'tol': 0.001,  # DEC时优化容忍度
                    'num_classes': 10,  # 数据集类别，MNIST为10
                    'img_size': (28, 28),  # 图像大小，MNIST为[28, 28]
                    'patch_size': (4, 4),  #
                    'in_chans': 1,  # 输入图像channel，MNIST为1
                    'embed_dim': 16,  # 编码大小，指定为p**2*channel
                    'depth': 2,  # 执行多少次Transformer
                    'num_heads': 4,  # 多头注意力中的“头”
                    'dim_head': 4,  # 多头注意力中每个“头”的维度
                    'decoder_embed_dim': 16,  # 用于计算decoder时的位置嵌入
                    'mlp_ratio': 4,  # Transformer中隐藏层维度倍数
                    'norm_pix_loss': False,  # 是否执行归一化
                    'alpha': 1,  # dddd
                    'n_clusters': 10,
                    'n_init': 20,
                    'interval': 1  # DEC时的训练间隔
                }
            },
            'Cifar10': {
                'CAE': {
                    'model': 'CAE',  # 做模型标记，后续用作聚类判断
                    'in_channels': 3,  # 在彩色图像如Cifar上Channel为3
                    'batch_size': 256,  # 预训练时的batch_size
                    'lr': 0.001,  # 预训练时的学习率
                    'alpha': 1.0,  # DEC时的alpha
                    'n_epochs': 100,  # 预选连的epoch数
                    'tol': 0.001,  # DEC时优化容忍度
                    'update_interval': 1,  # DEC时的训练间隔
                    'num_classes': 10,  # 在MNIST上
                    'basic_num': 2,  # 决定basicblock上的卷积次数，至少为1
                    'conv1_outplanes': 32,  # 初始卷积参数
                    'bolck1_outplanes': 64,
                    'bolck2_outplanes': 128,
                    'bolck3_outplanes': 256,
                    'bolck4_outplanes': 512,
                    'layers_num': 4,  # layers的层数，至少为1
                    'maxpool_dr': 1,  # 0表示不用maxpooling降维
                    'pool_bool': 0,  # 0表示不pooling，1表示avg，2表示max
                    'n_init': 20,  # kmeans的初始参数
                    'interval': 1  # DEC时的训练间隔
                }
            }
        }
        self.dataset = dataset
        self.model = model
        self.config = config[dataset][model]
    def CAE_n_z(self, dataset):
        if dataset == 'MNIST':
            if self.config['layers_num'] == 1:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck1_outplanes']*7*7
                else: n_z = self.config['bolck1_outplanes']
            elif self.config['layers_num'] == 2:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck2_outplanes']*4*4
                else: n_z = self.config['bolck2_outplanes']
            elif self.config['layers_num'] == 3:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck3_outplanes']*2*2
                else: n_z = self.config['bolck3_outplanes']
            elif self.config['layers_num'] == 4:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck4_outplanes']*1*1
                else: n_z = self.config['bolck4_outplanes']

        if dataset =='Cifar10':
            if self.config['layers_num'] == 1:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck1_outplanes']*8*8
                else: n_z = self.config['bolck1_outplanes']
            elif self.config['layers_num'] == 2:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck2_outplanes']*4*4
                else: n_z = self.config['bolck2_outplanes']
            elif self.config['layers_num'] == 3:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck3_outplanes']*2*2
                else: n_z = self.config['bolck3_outplanes']
            elif self.config['layers_num'] == 4:
                if self.config['pool_bool'] == 0:
                    n_z = self.config['bolck4_outplanes']*1*1
                else: n_z = self.config['bolck4_outplanes']
        return n_z

    def get_parameters(self):
        config_update = self.config
        if self.model == 'CAE':
            n_z = self.CAE_n_z(self.dataset)
            config_update.update({'n_z': n_z})
        return config_update
