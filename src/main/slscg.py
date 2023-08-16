import torch
import driver
from dataset import MnistDataset
from evalution import tsne_print
from DEC import DEC
from _config import Config
import pandas as pd
import numpy as np
import json
import os

config = Config(dataset='MNIST', model='MAE').get_parameters()
num = 18000

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset(num)

model = driver.model('MNIST', 'MAE', config)
model.to(device)
model.pretrain(dataset, batch_size=config['batch_size'], lr=config['lr'])

y_pred, acc_train, ari_train, nmi_train = DEC(model, dataset, config)

eval = np.array([acc_train, ari_train, nmi_train])
eval = eval.T
eval = pd.DataFrame(eval, columns = ['acc', 'ari', 'nmi'])

# creat log folder
path = "log/acc{}".format(eval['acc'].max())
if not os.path.exists(path):
    os.mkdir(path)
tsne_print(model, dataset, num, path)

eval.to_csv('{}eval.txt'.format(path),sep=' ')
with open('{}eval.txt'.format(path), 'a', encoding='utf-8') as f:
    f.write(json.dumps(config))
