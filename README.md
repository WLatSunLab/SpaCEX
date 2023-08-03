# Description
The repository are designed to facilitate easy addition, calling, and debugging of encoders all members of the TBD project team.
# input_data.py
input_data.py is used to load the MNIST dataset.
```shell
from input_data import MnistDataset

num=10000

dataset = MnistDataset(num)
```
# driver.py
This project stores different encoder methods such as CAE, VIT in the Encoder folder. To facilitate testing of different encoders, driver is used for encoder selection. If members need to add new encoders, please write based on the frameworks of existing encoders.
```shell
import driver
import torch
from input_data import MnistDataset

num=10000

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset(num)

model = driver.model('CAE')
model.to(device)
model.pretrain(dataset)

data = dataset.x
y = dataset.y
data = torch.Tensor(data).to(device)
data=data.unsqueeze(1)
x_bar, hidden, q = model(data)
```
# clustering.py
In this file, it is required to first pre-train the specified encoder and then perform DEC.
```shell
import torch
import driver
import torch
from input_data import MnistDataset
from clustering import clustering

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset(10000)

model = driver.model('CAE')
model.to(device)
model.pretrain(dataset)

y_pred = clustering(model, dataset)
```
