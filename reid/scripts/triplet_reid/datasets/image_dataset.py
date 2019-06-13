from datasets.dataset import Dataset


class FullImageDataset(Dataset):
    def __init__(self, make_dataset_fn, source_file, data_dir, name, *args, dataset_args={}, **kwargs):
        data, header, dataset_info = make_dataset_fn(source_file, data_dir, name, **dataset_args)
        super().__init__(data, header, dataset_info, name, *args, **kwargs)

    def __getitem__(self, index):
        data = self.data[index]
        copied = data.copy()
        img = self.loader_fn(copied['path'])
        if self.transform is not None:
            self.transform.to_deterministic()
            img = self.transform.augment_image(img)

        copied['img'] = img
        return copied
