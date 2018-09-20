import functools

from segcore.loss.loss import CrossEntropy2d
from segcore.loss.loss import cross_entropy2d
from segcore.loss.loss import bootstrapped_cross_entropy2d
from segcore.loss.loss import multi_scale_cross_entropy2d



loss_type = {'cross_entropy_ori': CrossEntropy2d,
             'cross_entropy': cross_entropy2d,
             'bootstrapped_cross_entropy': bootstrapped_cross_entropy2d,
             'multi_scale_cross_entropy': multi_scale_cross_entropy2d,}

def get_loss(cfg):
    if cfg['training']['loss'] is None:
        print("Using default cross entropy loss")
        return cross_entropy2d

    else:
        loss_dict = cfg['training']['loss']
        loss_name = loss_dict['name']
        loss_params = {k:v for k,v in loss_dict.items() if k != 'name'}

        if loss_name not in loss_type:
            raise NotImplementedError('Loss {} not implemented'.format(loss_name))

        print('Using {} with {} params'.format(loss_name,
                                                     loss_params))
        return functools.partial(loss_type[loss_name], **loss_params)
