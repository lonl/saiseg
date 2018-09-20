import json

from segcore.loader.ISPRS_Vaihingen import ISPRSVaihingenLoader



def get_loader(name):
    """get_loader

    :param name:
    """
    return {
        "isprs": ISPRSVaihingenLoader,
    }[name]

