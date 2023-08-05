

class Config:
    def __init__(self, model='CAE', dataset='MNIST'):
        config = {
            'CAE': {'batch_size': 256,  # 预训练时的batch_size
                    'lr': 0.001,  # 预训练时的学习率
                    'alpha': 1.0,  # DEC时的alpha
                    'n_epochs': 100,  # 预选连的epoch数
                    'tol': 0.001,  # DEC时优化容忍度
                    'update_interval': 1,  # DEC时的训练间隔
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
            'VIT':{},
            'MAE':{}
        }        
        self.model = 'CAE'
        self.config = config[model]
    def CAE_n_z(self):
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
        elif self.config['layers_num'] == 2:
            if self.config['pool_bool'] == 0:
                n_z = self.config['bolck4_outplanes']*1*1
            else: n_z = self.config['bolck4_outplanes']
        return n_z

    def get_parameters(self):
        config_update = self.config
        n_z = self.CAE_n_z()
        config_update.update({'n_z': n_z})
        return config_update