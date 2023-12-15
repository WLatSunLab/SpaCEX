

class Config:
    def __init__(self,  dataset='MNIST', model='MAE'):
        config = {
            'Mouse image': {
                'MAE': {
                    'decoder': 'Mouse image',
                    'model': 'MAE',  # used for clustering judgment
                    'batch_size': 256,  # batch_size in pre-training
                    'lr': 0.001,  # leraning rate in pre-training
                    'n_epochs': 10,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'num_classes': 400,  # category, dlpfc is 500
                    'img_size': (35, 30),  # image size，dlpfc is [28, 28]
                    'patch_size': (4, 4),  # patch size, dlpfc is [4, 4]
                    'in_chans': 1,  # image channel, dlpfc is 1
                    'embed_dim': 16,  # embedding dim，p**2*channel
                    'embed_dim_out': 64,
                    'depth': 5,  # number of transformer
                    'num_heads': 4,  # multi-head
                    'dim_head': 4,  # head dim in multi-head attention
                    'decoder_embed_dim': 16,  # positional embedding dim
                    'mlp_ratio': 5,  # ratio between the hidden dimension and the embedded dimension
                    'norm_pix_loss': False,  # wheather to normalize
                    'alpha': 0.8,  # alpha in DEC
                    'n_clusters': 400,  # number of category, dlpfc is 500
                    'n_init': 30,  # kmeans
                    'interval': 1,  # interval in DEC
                    'gamma': 0.1,  # coefficient of clustering loss
                    'l1': 0.1,  # coefficient of nega-likelihood loss
                    'l2': 0.1,  # coefficient of nega-seperate loss
                    'l3': 0.1,  # coefficient of nega-seperate loss
                    'l4': 0.1,  # coefficient of nega-size loss
                    'l5': 0.1,  # coefficient of sigma loss
                    'l6': 1  # coefficient of reconstrctive loss

                }
            },
            'Gene image': {
                'MAE': {
                    'decoder': 'Gene',
                    'model': 'MAE',  # used for clustering judgment
                    'batch_size': 256,  # batch_size in pre-training
                    'lr': 0.001,  # leraning rate in pre-training
                    'n_epochs': 10,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'num_classes': 500,  # category, dlpfc is 500
                    'img_size': (72, 59),  # image size，dlpfc is [28, 28]
                    'patch_size': (4, 4),  # patch size, dlpfc is [4, 4]
                    'in_chans': 1,  # image channel, dlpfc is 1
                    'embed_dim': 16,  # embedding dim，p**2*channel
                    'embed_dim_out': 64,
                    'depth': 5,  # number of transformer
                    'num_heads': 4,  # multi-head
                    'dim_head': 4,  # head dim in multi-head attention
                    'decoder_embed_dim': 16,  # positional embedding dim
                    'mlp_ratio': 5,  # ratio between the hidden dimension and the embedded dimension
                    'norm_pix_loss': False,  # wheather to normalize
                    'alpha': 0.8,  # alpha in DEC
                    'n_clusters': 500,  # number of category, dlpfc is 500
                    'n_init': 30,  # kmeans
                    'interval': 1,  # interval in DEC
                    'gamma': 0.1,  # coefficient of clustering loss
                    'l1': 0.1,  # coefficient of nega-likelihood loss
                    'l2': 0.1,  # coefficient of nega-seperate loss
                    'l3': 0.1,  # coefficient of nega-seperate loss
                    'l4': 0.1,  # coefficient of nega-size loss
                    'l5': 0.1,  # coefficient of sigma loss
                    'l6': 1  # coefficient of reconstrctive loss

                }
            },
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
                    'pool_bool': 0,  # 0 denote w/o pooling，1 denote avg，2 denote max
                    'n_init': 20,  # kmeans
                    'interval': 1  # interval in DEC
                },
                'VIT': {},
                'MAE': {
                    'decoder': 'MNIST', 
                    'model': 'MAE',  # used for clustering judgment
                    'batch_size': 128,  # batch_size in pre-training
                    'lr': 0.001,  # leraning rate in pre-training
                    'n_epochs': 500,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'num_classes': 10,  # category, MNIST is 10
                    'img_size': (28, 28),  # image size，MNIST is [28, 28]
                    'patch_size': (4, 4),  # patch size, MNIST is [4, 4]
                    'in_chans': 1,  # image channel, MNIST is 1
                    'embed_dim': 16,  # embedding dim，p**2*channel
                    'depth': 5,  # number of transformer
                    'num_heads': 4,  # multi-head
                    'embed_dim_out': 16, 
                    'dim_head': 4,  # head dim in multi-head attention
                    'decoder_embed_dim': 16,  # positional embedding dim
                    'mlp_ratio': 5,  # ratio between the hidden dimension and the embedded dimension
                    'norm_pix_loss': False,  # wheather to normalize
                    'alpha': 0.8,  # alpha in DEC
                    'n_clusters': 500, # number of category, MNIST is 10
                    'n_init': 20, # kmeans
                    'gamma': 0.1,
                    'interval': 1  # interval in DEC
                },
                'VGG':{
                    'model': 'VGG',  # used for clustering judgement
                    'batch_size': 256,  # batch size in pre-training
                    'lr': 0.001,  # learning rate in pre-training
                    'n_epochs': 100,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'num_classes': 10,  # number of category, MNIST is 10
                    'conv1_outplanes': 64,  # initial convolution
                    'conv2_outplanes': 128,
                    'conv3_outplanes': 256,
                    'conv4_outplanes': 512,
                    'hidden_size': 512,  # hidden dim
                    'p': 0.5,  # rate of dropout
                    'alpha': 1.0,  # alpha in DEC
                    'interval': 1  # interval in DEC
                },
                'SwinTransformer':{
                    'model': 'SwinTransformer',  # used for clustering judgement
                    'batch_size': 256,  # batch size in pre-training
                    'lr': 0.001,  # learning rate in pre-training
                    'n_epochs': 100,  # epoch in pre-training
                    'tol': 0.001,  # optimize tolerance in DEC
                    'num_classes': 10,  # number of category, MNIST is 10
                    'img_size': 28, # image size
                    'patch_size':2, # patch size
                    'n_channels':1, # channel of image
                    'embed_dim':48, # embedding dim
                    'window_size':7, # window attention size
                    'mlp_ratio':4, #  ratio between the hidden dimension and the embedded dimension
                    'drop_rate':0., # rate of dropout
                    'attn_drop_rate':0., # rate of dropout in attention
                    'n_swin_blocks':(2, 2), # number of swin tranformer in each layer
                    'n_attn_heads':(2, 4), # number of attention head in each layer
                    'alpha': 1.0,  # alpha in DEC
                    'n_z':96
                }
            },
            'Cifar10': {
                'CAE': {
                    'model': 'CAE',  # used for clustering judgement
                    'in_channels': 3,  # channel of image, Cifar10 is 3
                    'batch_size': 256,  # batch size in pre-training
                    'lr': 0.001,  # learning rate in pre-training
                    'n_epochs': 100,  # dpoch in pre-training
                    'tol': 0.001,  # oprimize tolerance in DEC
                    'num_classes': 10,  # number of category, Cifar10 is 10
                    'basic_num': 2,  # number of covolutions in basicblokc
                    'conv1_outplanes': 32,  # initial convolution
                    'bolck1_outplanes': 64,
                    'bolck2_outplanes': 128,
                    'bolck3_outplanes': 256,
                    'bolck4_outplanes': 512,
                    'layers_num': 4,  # number of basicblock
                    'maxpool_dr': 1,  # wheather to maxpooling
                    'pool_bool': 0,  # 0 denote w/o pooling，1 denote avg，2 denote max
                    'n_init': 20,  # kmeans
                    'alpha': 1.0,  # alpha in DEC
                    'interval': 1  # interval in DEC
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
