normalization = {
    "mean": [0.485, 0.456, 0.406],
    "std" : [0.229, 0.224, 0.225]
}


pose_transform_train = {
    "affinewithcrop": {
        "translate_percent": [-0.2, 0.2],
        "rotate": [-30, 30],
        "scale": [0.75, 1.25]
    },
    "fliplrwithpairs": {"p": 0.5},
    "resize": {
        "width": 256,
        "height": 256
    },
    "normalization": normalization
}


pose_transform_test = {
    "cropfrombbox": {},
    "resize": {
        "width": 256,
        "height": 256
    },
    "normalization": normalization
}


reid_transform_train = {
    "fliplr": {"p": 0.5},
    "resize": {
        "width": 128,
        "height": 256
    },
    "normalization": normalization
}


augmentation = ["HorizontalFlip"]

reid_transform_test = {
    "backend": "torchvision",
    "resize": {
        "width": 128,
        "height": 256
    },
    "normalization": normalization,
    "augmentations": augmentation
}
