from .transform import pose_transform_train, pose_transform_test, reid_transform_train, reid_transform_test

mpii_train_dataset = {
    "name": "mpii",
    "loader_fn": "cv2",
    "transform": pose_transform_train,
    "split": "train"
}


mpii_val_dataset = {
    "name": "mpii",
    "split": "val",
    "loader_fn": "cv2",
    "transform": pose_transform_test
}

market_train_dataset = {
    "name": "market1501",
    "transform": reid_transform_train,
}

market_test_dataset = {
    "name": "market1501_test",
    "loader_fn": "cv2pil",
    "transform": reid_transform_test,
    "metric": "euclidean"
}


duke_train_dataset = {
    "name": "duke",
    "transform": reid_transform_train,
    "type": "reid"
}

duke_test_dataset = {
    "name": "duke_test",
    "transform": reid_transform_test,
    "loader_fn": "cv2pil",
    "metric": "euclidean"
}
