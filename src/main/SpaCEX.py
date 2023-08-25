import torch
import torch.nn as nn
from SpaCEX.src.main import driver
from SpaCEX.src.main.clustering.DEC import DEC
from SpaCEX.src.main._config import Config
import pandas as pd
import numpy as np
import json
import os


class SpaCEX():
    def train(dataset, total):
        config = Config(dataset='Gene image', model='MAE').get_parameters()
        cuda = torch.cuda.is_available()
        print("use cuda: {}".format(cuda))
        device = torch.device("cuda" if cuda else "cpu")
        model = driver.model('Gene image', 'MAE', config)
        model.to(device)
        model.pretrain(dataset, batch_size=config['batch_size'], lr=config['lr'])
        y_pred, embedding= DEC(model, dataset, total, config)

        return model, y_pred, embedding

#def SpaCEX(dataset, total):
   # config = Config(dataset='Gene image', model='MAE').get_parameters()
   # cuda = torch.cuda.is_available()
   # print("use cuda: {}".format(cuda))
   # device = torch.device("cuda" if cuda else "cpu")
   # model = driver.model('Gene image', 'MAE', config)
   # model.to(device)
   # model.pretrain(dataset, batch_size=config['batch_size'], lr=config['lr'])
   # y_pred, embedding= DEC(model, dataset, total, config)

   # return model, y_pred, embedding
