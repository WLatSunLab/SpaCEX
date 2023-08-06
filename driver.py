from Encoder.CAE_encoder import CAE
#from Encoder.VIT_encoder import VIT
#from _config import Config

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


def model(dataset, encoder, config):
    if dataset =='MNIST':
        if encoder == 'CAE':
            model = CAE(
                in_channels=config['in_channels'],
                basic_num=config['basic_num'],
                conv1_outplanes=config['conv1_outplanes'],
                bolck1_outplanes=config['bolck1_outplanes'],
                bolck2_outplanes=config['bolck2_outplanes'],
                bolck3_outplanes=config['bolck3_outplanes'],
                bolck4_outplanes=config['bolck4_outplanes'],
                layers_num=config['layers_num'],
                maxpool_dr=config['maxpool_dr'],
                pool_bool=config['pool_bool'],
                alpha=config['alpha'],
                n_z = config['n_z'])
        #if encoder == 'VIT':
            #model = VIT()
    if dataset =='Cifar10':
        if encoder == 'CAE':
            model = CAE(
                in_channels=config['in_channels'],
                basic_num=config['basic_num'],
                conv1_outplanes=config['conv1_outplanes'],
                bolck1_outplanes=config['bolck1_outplanes'],
                bolck2_outplanes=config['bolck2_outplanes'],
                bolck3_outplanes=config['bolck3_outplanes'],
                bolck4_outplanes=config['bolck4_outplanes'],
                layers_num=config['layers_num'],
                maxpool_dr=config['maxpool_dr'],
                pool_bool=config['pool_bool'],
                alpha=config['alpha'],
                n_z = config['n_z'])
    return model

