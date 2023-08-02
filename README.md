# Description
The repository are designed to facilitate easy addition, calling, and debugging of encoders all members of the IDEC project team.
# input_data.py
input_data.py is used to load the MNIST dataset.
```shell
from input_data import MnistDataset
dataset = MnistDataset()
```
# driver.py
This project stores different encoder methods such as CAE, VIT in the Encoder folder. To facilitate testing of different encoders, driver is used for encoder selection. If members need to add new encoders, please write based on the frameworks of existing encoders.
```shell
from input_data import MnistDataset
import driver
import torch

cuda = torch.cuda.is_available()
print("use cuda: {}".format(cuda))
device = torch.device("cuda" if cuda else "cpu")
dataset = MnistDataset()

model = driver.model('CAE')
model.to(device)
model.pretrain(dataset)
```
