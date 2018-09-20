from torch.optim import SGD
from torch.optim import Adam
from torch.optim import ASGD
from torch.optim import Adamax
from torch.optim import Adadelta
from torch.optim import Adagrad
from torch.optim import RMSprop


opt_type =  {'sgd': SGD,
             'adam': Adam,
             'asgd': ASGD,
             'adamax': Adamax,
             'adadelta': Adadelta,
             'adagrad': Adagrad,
             'rmsprop': RMSprop,}

def get_optimizer(cfg):
    if cfg['training']['optimizer'] is None:
        print("Using SGD optimizer")
        return SGD

    else:
        opt_name = cfg['training']['optimizer']['name']
        if opt_name not in opt_type:
            raise NotImplementedError('Optimizer {} not implemented'.format(opt_name))

        print('Using {} optimizer'.format(opt_name))
        return opt_type[opt_name]

