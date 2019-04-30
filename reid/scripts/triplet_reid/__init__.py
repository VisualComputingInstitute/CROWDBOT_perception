from importlib import import_module

def build_cfg_fn(name):
    module = import_module('schedulers.scheduler')
    return getattr(module, '{}_cfg'.format(name))
