from SpaCEX.src.main.encoder.MAE_encoder import MAE


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


def model(dataset, encoder, config, image_size):
    if dataset == 'Gene image':
        if encoder =='MAE':
            model = MAE(
                decoder=config['decoder'],
                #img_size=config['img_size'],
                img_size = image_size,
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
                embed_dim_out = config['embed_dim_out']
            )
    if dataset == 'Mouse image':
        if encoder =='MAE':
            model = MAE(
                decoder=config['decoder'],
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
                embed_dim_out = config['embed_dim_out']
            )
    if dataset == 'MNIST':
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
                decoder=config['decoder'],
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

