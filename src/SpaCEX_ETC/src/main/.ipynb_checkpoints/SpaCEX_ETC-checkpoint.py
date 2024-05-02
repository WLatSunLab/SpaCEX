import torch
import torch.nn as nn
from SpaCEX.src.main import driver
from SpaCEX.src.main.SpaCEX_ETC.src.main.clustering.DEC import DEC
from SpaCEX.src.main.SpaCEX_ETC.src.main._config import Config
import pandas as pd
import numpy as np
import json
import os

'''
What you should input
dataset: gene image data with size [N, 1, 72, 59]
total: gene similarity matrix with size [N, N]

What you will get return
model: encoder that haved been trained
y_pred: label that SpaCEX generative
embedding: embedding that generatived by encoder

Others
If you wanna get other return such as x_bar or parameters of SMM, just rewrite DEC to get what you want.
'''

class SpaCEX():
    def train(dataset, dataset_denoise=None, total=None, pretrain=False):
        config = Config(dataset='Mouse image', model='MAE').get_parameters()
        cuda = torch.cuda.is_available()
        print("use cuda: {}".format(cuda))
        device = torch.device("cuda" if cuda else "cpu")
        model = driver.model('Mouse image', 'MAE', config)
        model.to(device)
        if config['decoder'] == 'Gene image':
            model.pretrain(dataset = dataset, 
                           dataset_denoise = dataset_denoise, 
                           batch_size=config['batch_size'], 
                           lr=config['lr'])
        else:
            model.pretrain(dataset = dataset,
                           dataset_denoise = dataset_denoise, 
                           batch_size=256, 
                           lr=1e-3,
                           pretrain=False)
        y_pred, z, model= DEC(model, dataset, total = total, config = config)

        return y_pred, model

