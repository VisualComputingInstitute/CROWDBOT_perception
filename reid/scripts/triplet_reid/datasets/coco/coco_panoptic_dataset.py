import json
import os
from datasets.utils import HeaderItem
from datasets.segmentation_dataset import SegmentationDataset

from datasets import register_dataset
from builders import transform_builder
from evaluation import Evaluation
import numpy as np
from logger import get_tensorboard_logger
import torch
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


def get_subdir_from_split_for_id(split):

    res = 'annotations/'
    if split == 'train':
        return res + 'panoptic_train2017'
    elif split == 'val':
        return res + 'panoptic_val2017'
    else:
        raise ValueError


def get_subdir_from_split_for_sem(split):

    res = 'annotations/'
    if split == 'train':
        return res + 'semantic_train2017'
    elif split == 'val':
        return res + 'semantic_val2017'
    else:
        raise ValueError


def get_path_to_categories():
    return 'annotations/panoptic_coco_categories.json'


def make_coco_panoptic(annotation_file_path, data_dir, split='train'):
    from .coco_panoptic_helper import COCOPanopticHelper

    coco = COCOPanopticHelper(annotation_file_path)
    image_ids = coco.getImgIds([])

    index_to_image = {k: v for k, v in enumerate(image_ids)}

    data = []

    subdir_input = get_subdir_from_split_for_input(split)
    subdir_id = get_subdir_from_split_for_id(split)
    subdir_sem = get_subdir_from_split_for_sem(split)

    ids_to_thing = dict()
    for cat in coco.cats.values():
        ids_to_thing[cat["id"]] = cat["isthing"] == 1

    for image_id in index_to_image.values():
        datum = dict()
        img_info = coco.loadImgs(image_id)
        file_name = img_info[0]['file_name']
        datum['path'] = data_dir + '/' + subdir_input + '/' + file_name
        datum['id-path'] = data_dir + '/' + subdir_id + '/' + file_name.replace('jpg', 'png')
        datum['sem-path'] = data_dir + '/' + subdir_sem + '/' + file_name.replace('jpg', 'png')
        datum['image_id'] = image_id
        datum['file_name'] = file_name.replace('jpg', 'png')

        segms = coco.loadAnns(image_id)[0]["segments_info"]
        datum['fgd_segms'] = [segm for segm in segms if ids_to_thing[segm["category_id"]]]


        data.append(datum)

    with open(data_dir + '/' + get_path_to_categories()) as fp:
        cats = json.load(fp)

    cat_ids = [entry['id'] for entry in cats]

    removed_list = []
    for i in range(np.min(cat_ids), np.max(cat_ids) + 1):
        if i in cat_ids:
            continue
        else:
            removed_list.append(i)

    conversion = np.full(256, -1)
    # ignore id = 255
    conversion[0] = 255

    orig_start = 1
    start = 0

    for removed_id in removed_list:
        orig_end = removed_id
        end = len(conversion[orig_start:orig_end]) + start

        conversion[orig_start:orig_end] = np.array(range(start, end))
        orig_start = orig_end + 1
        start = end

    conversion[orig_start:np.max(cat_ids) + 1] = np.array(range(start, np.max(cat_ids) - len(removed_list)))

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
            'num_classes': 133,
            'raw_classes': 256} #dirty hack

    print("Dataset COCO Panoptic loaded with %d images" % (len(data)))

    return (data, header, info)

@register_dataset("coco_panoptic")
class COCOPanopticDataset(SegmentationDataset):

    def __init__(self,  source_file, *args, **kwargs):

        self.source_file = source_file

        super().__init__("COCOPanopticDataset", *args, **kwargs)

    @staticmethod
    def build(cfg, *args, **kwargs):
        data_dir = Config.COCO_PANOPTIC_DATA
        split = cfg['split']
        if split == 'train':
            source_file = Config.COCO_PANOPTIC_SOURCE_TRAIN
        else:
            source_file = Config.COCO_PANOPTIC_SOURCE_VAL
        transform = transform_builder.build(cfg['transform'])
        data, header, info = make_coco_panoptic(source_file, data_dir, split)
        return COCOPanopticDataset(source_file, data, header, info, transform, *args, **kwargs)

    def get_evaluation(self, model):
        #return COCOPanopticSemanticEvaluation(self.source_file, model, self.info)
        return COCOPanopticEvaluation(self.source_file, self.info['num_classes'], self.info)

    def __getitem__(self, index):
        datum = self.data[index]
        copied = datum.copy()

        tmp = self.loader_fn(copied['path']).convert("RGB")
        img = np.array(tmp)
        tmp.close()
        orig_img = img.copy()
        tmp = self.loader_fn(copied['sem-path'])
        seg = np.array(tmp)
        tmp.close()
        bboxs = np.zeros((100, 5))
        segments = copied['fgd_segms']
        for idx, segm in enumerate(segments):
            my_list = [self.info['conversion'][segm['category_id']]]
            my_list.extend(segm['bbox'])
            bboxs[idx] = np.array(my_list)

        #ids = np.array(self.loader_fn(copied['id-path']).convert("RGB"))
        #ids = ids[..., 0] + ids[..., 1] * 256 + ids[..., 2] * 256 * 256
        if self.transform is not None:
            self.transform.to_deterministic()
            img, orig_img = self.transform.augment_image(img, return_unnormalized=True)
            seg = self.transform.augment_segmentation(seg, self.info['raw_classes'])

            # Ids can be HUGE so decrease them before augmenting due to conversion to heatmaps
            # ids_unique = np.unique(ids)
            # mapping = {c: d for c, d in zip(ids_unique, range(len(ids_unique)))}
            # inv_mapping = {d: c for c, d in mapping.items()}
            # ids_for_augment = np.zeros_like(ids)
            # for elem in ids_unique:
            #     ids_for_augment[ids == elem] = mapping[elem]
            # ids_for_augment = self.transform.augment_segmentation(ids_for_augment, len(ids_unique))
            # ids = np.zeros_like(ids_for_augment)
            # for elem in np.unique(ids_for_augment):
            #     ids[ids_for_augment == elem] = inv_mapping[elem]

        if 'conversion' in self.info:
             seg = self.info['conversion'][seg]
        copied['gt-seg'] = torch.from_numpy(seg)
        copied['gt-bboxs'] = torch.from_numpy(bboxs).to(torch.float32)
        copied['img'] = img
        copied['orig-img'] = orig_img
        return copied


from writers.coco import COCOJsonWriter
class COCOPanopticSemanticEvaluation(Evaluation):
    def __init__(self, source_file, model, dataset_info):
        super().__init__("COCOPanopticSemanticEvaluation")
        self.source_file = source_file
        self.dataset_info = dataset_info
        self.tensorboard_logger = get_tensorboard_logger()

    def before_saving(self, endpoints, data):

        gt_segs = data['gt-seg'].detach().cpu().numpy()
        imgs = data['orig-img']
        predictions = torch.argmax(endpoints['sem-logits'], dim=1).detach().cpu().numpy()

        for i in range(imgs.shape[0]):
            self.tensorboard_logger.add_image("image", np.transpose(imgs[i, :], (2, 0, 1)))
            self.tensorboard_logger.add_image("sem-gt", gt_segs[i, :], dataformat="HW")
            self.tensorboard_logger.add_image("sem-pred", predictions[i, :], dataformat="HW")

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
        from .coco_panoptic_helper import COCOPanopticEvalHelper
        #TODO: Make nice
        coco_gt = COCO("/globalwork/weber/COCO/annotations/panoptic_val2017_cocoformat.json")

        coco_dt = coco_gt.loadRes(self.output_file)
        cocoEval = COCOPanopticEvalHelper(coco_gt, coco_dt, stuffStartId=1, stuffEndId=200)
        cocoEval.evaluate()
        stats, statsPerClass = cocoEval.summarize()
        new_measures = dict()
        new_measures["mIoU"]        = stats[0]
        new_measures["fwIoU"]       = stats[1]
        new_measures["mAcc"]        = stats[2]
        new_measures["pAcc"]        = stats[3]
        #new_measures.update(statsPerClass)

        return new_measures

    def get_writer(self, output_path):
        self.output_path = output_path
        self.output_file = os.path.join(self.output_path, "segmentations.json")
        self.writer = COCOJsonWriter(self.output_file)
        return self.writer


from writers.coco import COCOPanopticJSONWriter
from metrics import fast_hist, calc_seg_score
from datasets.coco.coco_panoptic_eval import pq_compute
from settings import Config
class COCOPanopticEvaluation(Evaluation):
    def __init__(self, source_file, num_classes, dataset_info):
        super().__init__("COCOPanopticEvaluation")
        self.source_file = source_file
        self.dataset_info = dataset_info
        self.tensorboard_logger = get_tensorboard_logger()
        self.num_classes = num_classes
        self.hist_all = np.zeros((self.num_classes, self.num_classes))
        #self.hist_stuff = np.zeros((53, 53))
        #self.hist_things = np.zeros((80, 80))

    def before_saving(self, endpoints, data):

        gt_segs = data['gt-seg'].detach().cpu().numpy()
        imgs = data['orig-img']
        predicted_sem = endpoints['sem-pred'].detach().cpu().numpy()
        predicted_ids = endpoints['inst-pred'].detach().cpu().numpy()

        for i in range(imgs.shape[0]):
            self.hist_all += fast_hist(gt_segs[i, :].flatten(), predicted_sem[i, :].flatten(), self.num_classes)

            #gt_stuff = gt_segs[i, :].flatten() - 80
            #pred_stuff = predicted_sem[i, :].flatten() - 80
            #self.hist_stuff += fast_hist(gt_stuff, pred_stuff, 53)
            #self.hist_things += fast_hist(gt_segs[i, :].flatten(), predicted_sem[i, :].flatten(), 80)

            self.tensorboard_logger.add_image("image", np.transpose(imgs[i, :], (2, 0, 1)))
            self.tensorboard_logger.add_image("sem-gt", gt_segs[i, :], dataformat="HW")
            self.tensorboard_logger.add_image("sem-pred", predicted_sem[i, :], dataformat="HW")

        if 'inverse' in self.dataset_info:
            inverse_labels = self.dataset_info['inverse']
            inverse = True
        else:
            inverse_labels = None
            inverse = False

        data_to_write = {
            'predicted_sem': predicted_sem,
            'predicted_ids': predicted_ids,
            'file_names': data['file_name'],
            'ids': data['image_id'],
            'inverse_labels': inverse_labels,
            'inverse': inverse
        }

        return data_to_write

    def _rename_seg_scores(self, score):
        new_measures = dict()
        new_measures["mIoU"] = score['miou']
        new_measures["fwIoU"] = score['fwavacc']
        new_measures["mAcc"] = score['mean_accuracy']
        new_measures["pAcc"] = score['overall_accuracy']
        return new_measures

    def score(self):

        pq_scores = pq_compute(Config.COCO_PANOPTIC_SOURCE_VAL, self.output_file)

        score = calc_seg_score(self.hist_all)
        #score_stuff = calc_seg_score(self.hist_stuff)
        #score_things = calc_seg_score(self.hist_things)

        new_measures = self._rename_seg_scores(score)
        pq_scores["All"].update(new_measures)
        #pq_scores["Stuff"].update(self._rename_seg_scores(score_stuff))
        #pq_scores["Things"].update(self._rename_seg_scores(score_things))

        new_measures["pq"] = pq_scores["All"]["pq"]
        new_measures["rq"] = pq_scores["All"]["rq"]
        new_measures["sq"] = pq_scores["All"]["sq"]
        new_measures["pq_things"] = pq_scores["Things"]["pq"]
        new_measures["pq_stuff"] = pq_scores["Stuff"]["pq"]
        per_class_scores = dict()
        for entry in pq_scores["per_class"].keys():
            per_class_scores[str(entry)] = pq_scores["per_class"][entry]

        pq_scores["All"]["per_class"] = per_class_scores
        new_measures.update(pq_scores)
        #new_measures.update(statsPerClass)

        return new_measures

    def get_writer(self, output_path):
        self.output_path = output_path
        self.output_file = os.path.join(self.output_path, "panoptic-segmentations.json")
        self.writer = COCOPanopticJSONWriter(self.output_file)
        return self.writer

