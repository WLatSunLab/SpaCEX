import torch
from SpaCEX.src.main import driver
from SpaCEX.src.main.clustering.DEC import DEC
from SpaCEX.src.main._config import Config
import pandas as pd
import numpy as np
import json
import os

class SpaCEX():
  def __init__(self, dataset, total):
    super(SpaCEX, self).__init__()
    super(SpaCEX, self).__init__()
    self.config = Config(dataset='Gene image', model='MAE').get_parameters()
    self.dataset = dataset
    self.total = total
  def forward(self):
    cuda = torch.cuda.is_available()
    print("use cuda: {}".format(cuda))
    device = torch.device("cuda" if cuda else "cpu")
    model = driver.model('Gene image', 'MAE', self.config)
    model.to(device)
    model.pretrain(self.dataset, batch_size=self.config['batch_size'], lr=self.config['lr'])
    y_pred, embedding= DEC(model, self.dataset, self.total, self.config)

    return model, y_pred, embedding
