from CAE_encoder import CAE
from MAE_encoder import MAE
from VGG_encoder import VGG
from SwinTranformer_encoder import SwinTransformer


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
        if encoder == 'MAE':
            model = MAE(
                img_size=config['img_size'],
                patch_size=config['patch_size'],
                in_chans=config['in_chans'],
                embed_dim=config['embed_dim'],
                depth=config['depth'],
                num_heads=config['num_heads'],
                dim_head=config['dim_head'],
                decoder_embed_dim=config['decoder_embed_dim'],
                mlp_ratio=config['mlp_ratio'],
                norm_pix_loss=config['norm_pix_loss'],
                alpha=config['alpha'],
                n_clusters=config['n_clusters'],
            )
        if encoder == 'VGG':
            model = VGG(
                conv1_outplanes=config['conv1_outplanes'],
                conv2_outplanes=config['conv2_outplanes'],
                conv3_outplanes=config['conv3_outplanes'],
                conv4_outplanes=config['conv4_outplanes'],
                hidden_size=config['hidden_size'],
                p=config['p']
            )
        if encoder == 'SwinTransformer':
            model = SwinTransformer(
                img_size=config['img_size'],
                patch_size=config['patch_size'],
                in_channels=config['in_channels'],
                embed_dim=config['embed_dim'],
                window_size=config['window_size'],
                mlp_ratio=config['mlp_ratio'],
                drop_rate=config['drop_rate'],
                attn_drop_rate=config['attn_drop_rate'],
                n_swin_blocks=config['n_swin_blocks'],
                n_attn_heads=config['n_attn_heads'],
                n_z=config['n_z']
            )
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

