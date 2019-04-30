from datasets.utils import HeaderItem
from datasets.segmentation_dataset import SegmentationDataset
from datasets import register_dataset
import numpy as np
from builders import transform_builder
import torch
from logger import get_tensorboard_logger
from evaluation import Evaluation
import os
from settings import Config


def get_subdir_from_split_for_input(split):

    if split == 'train':
        return 'train2017'
    elif split == 'val':
        return 'val2017'
    elif split == 'test':
        return 'test2017'
    else:
        raise ValueError


def get_subdir_from_split_for_gt(split):

    res = 'annotations/'
    if split == 'train':
        return res + 'stuff_train2017'
    elif split == 'val':
        return res + 'stuff_val2017'
    else:
        raise ValueError

def make_coco_stuff(annotation_file_path, data_dir, split='train'):
    from pycocotools.coco import COCO

    coco = COCO(annotation_file_path)
    ann_ids = coco.getAnnIds([])
    image_ids = coco.getImgIds([])

    index_to_image = {k: v for k, v in enumerate(image_ids)}

    data = []

    subdir_input = get_subdir_from_split_for_input(split)
    subdir_gt = get_subdir_from_split_for_gt(split)

    for image_id in index_to_image.values():
        datum = dict()
        img_info = coco.loadImgs(image_id)
        file_name = img_info[0]['file_name']
        datum['path'] = data_dir + '/' + subdir_input + '/' + file_name
        datum['gt-path'] = data_dir + '/' + subdir_gt + '/' + file_name.replace('jpg', 'png')
        datum['image_id'] = image_id

        data.append(datum)

    conversion = np.full(256, -1)
    # ignore id = 255
    conversion[0] = 255
    # set stuff labels to range from 0 to 90
    conversion[92:183] = np.array(range(91))
    # set things_merged label to 91
    conversion[183] = 91

    inverse = np.full(256, -1)
    for i in range(len(conversion)):
        if conversion[i] == -1:
            continue
        inverse[conversion[i]] = i

    header = {'path': HeaderItem((), ""),
              'image_id': HeaderItem((), -1)
              }

    info = {'conversion': conversion,
            'inverse': inverse,
            'type': 'segmentation',
            'raw_classes': 256} #dirty hack

    print("Dataset COCO Stuff loaded with %d images" % (len(data)))

    return (data, header, info)


@register_dataset("coco_stuff")
class COCOStuffDataset(SegmentationDataset):

    def __init__(self, source_file, *args, **kwargs):
        self.source_file = source_file
        super().__init__("COCOStuffDataset", *args, **kwargs)

    @staticmethod
    def build(cfg, *args, **kwargs):
        data_dir = Config.COCO_STUFF_DATA
        split = cfg['split']
        if split == 'train':
            source_file = Config.COCO_STUFF_SOURCE_TRAIN
        else:
            source_file = Config.COCO_STUFF_SOURCE_VAL

        transform = transform_builder.build(cfg['transform'])
        data, header, info = make_coco_stuff(source_file, data_dir, split)
        return COCOStuffDataset(source_file, data, header, info, transform, *args, **kwargs)

    def get_evaluation(self, model):
        return COCOStuffEvaluation(self.source_file, model, self.info)


from writers.coco import COCOJsonWriter
class COCOStuffEvaluation(Evaluation):
    def __init__(self, source_file, model, dataset_info):
        super().__init__("COCOStuffEvaluation")
        self.source_file = source_file
        self.dataset_info = dataset_info
        self.tensorboard_logger = get_tensorboard_logger()
        #assert 'prediction' in self.keys

    def before_saving(self, endpoints, data):

        gt_segs = data['gt-seg'].detach().cpu().numpy()
        predictions = torch.argmax(endpoints['sem-logits'], dim=1).detach().cpu().numpy()
        imgs = data['orig-img']

        for i in range(imgs.shape[0]):
            self.tensorboard_logger.add_image("image", imgs[i, :])
            self.tensorboard_logger.add_image("ground-truth", gt_segs[i, :], dataformat="HW")
            self.tensorboard_logger.add_image("prediction", predictions[i, :], dataformat="HW")

        if 'inverse' in self.dataset_info:
            inverse_labels = self.dataset_info['inverse']
            inverse = True
        else:
            inverse_labels = None
            inverse = False

        data_to_write = {
            'predictions': predictions,
            'ids': data['image_id'],
            'inverse_labels': inverse_labels,
            'inverse': inverse
        }

        return data_to_write


    def score(self):
        from pycocotools.coco import COCO
        from pycocotools.cocostuffeval import COCOStuffeval

        coco_gt = COCO(self.source_file)

        coco_dt = coco_gt.loadRes(self.output_file)
        cocoEval = COCOStuffeval(coco_gt, coco_dt)
        cocoEval.evaluate()
        stats, statsPerClass = cocoEval.summarize()
        new_measures = dict()
        new_measures["mIoU"]        = stats[0]
        new_measures["fwIoU"]       = stats[1]
        new_measures["mAcc"]        = stats[2]
        new_measures["pAcc"]        = stats[3]
        new_measures["mIoUSup"]     = stats[4]
        new_measures["fwIoUSup"]    = stats[5]
        new_measures["mAccSp"]      = stats[6]
        new_measures["pAccSup"]     = stats[7]
        #new_measures.update(statsPerClass)

        return new_measures

    def get_writer(self, output_path):
        self.output_path = output_path
        self.output_file = os.path.join(self.output_path, "segmentations.json")
        self.writer = COCOJsonWriter(self.output_file)
        return self.writer
