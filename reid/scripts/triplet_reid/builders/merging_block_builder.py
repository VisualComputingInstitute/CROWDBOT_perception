from models.merging_blocks import SingleBlock, ConcatBlock

def build(cfg):
    name = cfg['name'].lower()
    if name == 'single':
        endpoint = cfg['endpoint']
        return SingleBlock(endpoint)
    elif name == 'concat':
        endpoints = cfg['endpoints']
        assert isinstance(endpoints, list)
        return ConcatBlock(endpoints)

    raise ValueError

