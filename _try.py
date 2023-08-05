import torch
import driver
from input_data import MnistDataset
from clustering import clustering, tsne_print
from _config import Config
import pandas as pd
import numpy as np
import json
import os

config = Config().get_parameters()
num = 10000

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset(num)

model = driver.model('CAE',config)
model.to(device)
model.pretrain(dataset, batch_size=config['batch_size'], path='/content/cae_mnist.pkl')

y_pred, acc_train, ari_train, nmi_train = clustering(model, dataset, config)

eval = np.array([acc_train, ari_train, nmi_train])
eval = eval.T
eval = pd.DataFrame(eval, columns = ['acc', 'ari', 'nmi'])
# 创建的目录
path = "log/acc{}".format(eval['acc'].max())
if not os.path.exists(path):
    os.mkdir(path)
tsne_print(model, dataset, num, path)

eval.to_csv('{}eval.txt'.format(path),sep=' ')
with open('{}eval.txt'.format(path), 'a', encoding='utf-8') as f:
    # 将dic dumps json 格式进行写入
    f.write(json.dumps(config))
