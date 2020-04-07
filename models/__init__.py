from .unet import *

models = {
    'unet' : get_unet
}
def get_model(name, **kwargs):
    return models[name.lower()](**kwargs)