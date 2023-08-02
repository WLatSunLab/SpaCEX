from CAE_encoder import CAE
from VIT_encoder import VIT
from torch.utils.data import DataLoader

'''
driver.py is used for DataLoader and selection models
'''
#def dataload(dataset):
    #train_loader = DataLoader(dataset, batch_size=256, shuffle=True)
    #return train_loader

'''
Pass character type arguments as 'CAE'
model = model('CAE')
'''
def model(encoder):
    if encoder == 'CAE':
        model = CAE()
    if encoder == 'VIT':
        model = VIT()
    return model

#def pretrain(model, dataset):
    #model.pretrain(dataset)

