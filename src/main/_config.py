

class Config:
    def __init__(self,  dataset='MNIST', model='MAE'):
        config = {
            'MNIST': {
                'CAE': {
                    'model': 'CAE',  # used for clustering judgment
                    'in_channels': 1,  # MNIST
                    'batch_size': 256,  # batch_size in pre-training
                    'lr': 0.001,  # learning rate in pre-training
                    'alpha': 1.0,  # alpha in DEC
                    'n_epochs': 20,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'num_classes': 10,  # MNIST
                    'basic_num': 1,  # number of convolutions
                    'conv1_outplanes': 32,  # initial convolution
                    'bolck1_outplanes': 64,
                    'bolck2_outplanes': 128,
                    'bolck3_outplanes': 256,
                    'bolck4_outplanes': 512,
                    'layers_num': 2,  # number of basicblock
                    'maxpool_dr': 1,  # Whether to use maxpooling
                    'pool_bool': 0,  # 0 denote pooling，1 denote avg，2denote max
                    'n_init': 20,  # kmeans
                    'interval': 1  # interval in DEC
                },
                'VIT': {},
                'MAE': {
                    'model': 'MAE',  # used for clustering judgment
                    'batch_size': 256,  # batch_size in pre-training
                    'lr': 0.01,  # leraning rate in pre-training
                    'n_epochs': 20,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'num_classes': 10,  # category, MNIST is 10
                    'img_size': (28, 28),  # image size，MNIST is [28, 28]
                    'patch_size': (4, 4),  # patch size, MNIST is [4, 4]
                    'in_chans': 1,  # image channel, MNIST is 1
                    'embed_dim': 16,  # embedding dim，p**2*channel
                    'depth': 5,  # number of transformer
                    'num_heads': 4,  # multi-head
                    'dim_head': 4,  # head dim in multi-head attention
                    'decoder_embed_dim': 16,  # positional embedding dim
                    'mlp_ratio': 5,  # mlp_ratio*hidden dim in transformer
                    'norm_pix_loss': False,  # wheather to normalize
                    'alpha': 0.8,  # alpha in DEC
                    'n_clusters': 10, # number of category, MNIST is 10
                    'n_init': 20, # kmeans
                    'interval': 1  # interval in DEC
                },
                'VGG':{
                    'model': 'VGG',  # used for clustering judgement
                    'batch_size': 256,  # batch size in pre-training
                    'lr': 0.001,  # learning rate in pre-training
                    'n_epochs': 100,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'update_interval': 1,  # interval in DEC
                    'num_classes': 10,  # number of category, MNIST is 10
                    'conv1_outplanes': 64,  # initial convolution
                    'conv2_outplanes': 128,
                    'conv3_outplanes': 256,
                    'conv4_outplanes': 512,
                    'hidden_size':512,  # hidden dim
                    'p': 0.5,  # rate of dropout
                    'alpha': 1.0,  # alpha in DEC
                    'interval': 1  # interbal
                }
                'SwinTransformer':{
                    'model': 'SwinTransformer',  # 做模型标记，后续用作聚类判断
                    'batch_size': 256,  # 预训练时的batch_size
                    'lr': 0.001,  # 预训练时的学习率
                    'alpha': 1.0,  # DEC时的alpha
                    'n_epochs': 100,  # 预选连的epoch数
                    'tol': 0.001,  # DEC时优化容忍度
                    'update_interval': 1,  # DEC时的训练间隔
                    'num_classes': 10,  # 在MNIST上
                    'img_size:28, # MNIST图像大小
                    'patch_size':2, # 输入图像分割成小块（patch）的大小
                    'n_channels':1, # 输入图像的通道数
                    'embed_dim':48, # 嵌入维度
                    'window_size':7, # 局部注意力窗口的大小
                    'mlp_ratio':4, #  MLP（多层感知机）部分中隐藏层的维度与嵌入维度之间的比率
                    'drop_rate':0., # Dropout的概率
                    'attn_drop_rate':0., # 注意力分数（权重）的Dropout概率
                    'n_swin_blocks':(2, 2), # 每个层级的Swin Transformer块的数量
                    'n_attn_heads':(2, 4), # 每个层级的注意力头的数量
                    'n_z':96
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
