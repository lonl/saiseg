import copy
import torchvision.models as models


from segcore.models.segnet import *
from segcore.models.unet import *
from segcore.models.linknet import *



def get_model(name, n_classes, version=None):

    model = _get_model_instance(name)

    if name == "segnet":
        #model = model(n_classes=n_classes, **param_dict)
        #vgg16 = models.vgg16(pretrained=True)
        #model.init_vgg16_params(vgg16)
        pass
    if name == "unet":

        pass

    if name == "linknet":

        pass



    return model


def _get_model_instance(name):
    try:
        return {
            "segnet": SegNet,
            "unet": Unet,
            "linknet": Linknet
        }[name]
    except:
        raise("Model {} not available".format(name))
