"""
There are multiple folders. One for LIP (What we want), one for Fashion Design (Ac), and one for multiple people (CIHP)


Folder Structure
Testing_images/Testing_images/testing_images: Test images
TrainVal_images/TrainVal_images/train_images: train images, ignore text files
TrainVal_images/TrainVal_images/val_images: train images, ignore text files
TrainVal_parsing_annotations/TrainVal_images/train_images: Train segmetation map
TrainVal_parsing_annotations/TrainVal_images/val_images: Val segmentation map
TrainVal_pose_annotations: json files of pose annotation

from source with caching.

"""
import pandas as pd
from logger import get_logger
import os
from .pose_dataset import JointInfo
from datasets import register_dataset
from datasets.utils import HeaderItem
from datasets.pose_dataset import PoseDataset
from builders import transform_builder
import numpy as np
from settings import Config
from evaluation import Evaluation
import torch
from writers.dummy import DummyWriter
from writers.memory import MemoryWriter
from utils import cache_result_on_disk
from metrics import calculate_pckh
from metrics import calc_seg_score
from transforms.flip_lr_with_pairs import FliplrWithPairs
import imgaug as ia
from metrics import fast_hist


def make_joint_info():
    short_names = [
        'r_ank', 'r_kne', 'r_hip', 'l_hip', 'l_kne', 'l_ank', 'b_pelv', 'b_spine',
        'b_neck', 'b_head', 'r_wri', 'r_elb', 'r_sho', 'l_sho', 'l_elb', 'l_wri']

    full_names = [
        'right ankle', 'right knee', 'right hip', 'left hip', 'left knee',
        'left ankle', 'pelvis', 'spine', 'neck', 'head', 'right wrist',
        'right elbow', 'right shoulder', 'left shoulder', 'left elbow',
        'left wrist']

    joint_info = JointInfo(short_names, full_names)
    j = joint_info.ids
    joint_info.stick_figure_edges = [
        (j.l_sho, j.l_elb), (j.r_sho, j.r_elb), (j.l_elb, j.l_wri),
        (j.r_elb, j.r_wri), (j.l_hip, j.l_kne), (j.r_hip, j.r_kne),
        (j.l_kne, j.l_ank), (j.r_kne, j.r_ank), (j.b_neck, j.b_head),
        (j.b_pelv, j.b_spine)]
    return joint_info

CLASSES = {
    0: "Background",
    1: "Hat",
    2: "Hair",
    3: "Glove",
    4: "Sunglasses",
    5: "UpperClothes",
    6: "Dress",
    7: "Coat",
    8: "Socks",
    9: "Pants",
    10: "Jumpsuits",
    11: "Scarf",
    12: "Skirt",
    13: "Face",
    14: "Left-arm",
    15: "Right-arm",
    16: "Left-leg",
    17: "Right-leg",
    18: "Left-shoe",
    19: "Right-shoe"
}

class SegInfo(object):
    # pickle does not like namedtuple
    def __init__(self, id_to_label, pairs):
        self.id_to_label = id_to_label
        self.pairs = pairs

def make_seg_info():
    id_to_label = CLASSES
    label_to_id = {value: key for key, value in id_to_label.items()}
    def build_pairs(label_to_id):
        pairs = dict()
        for label in label_to_id:
            if label.startswith('Left'):
                pair1 = label_to_id[label]
                label2 = 'Right' + label[len('Left'):]
                pair2 = label_to_id[label2]
            elif label.startswith('Right'):
                pair1 = label_to_id[label]
                label2 = 'Left' + label[len('Right'):]
                pair2 = label_to_id[label2]
            else:
                continue
            pairs[pair1] = pair2
        return pairs

    pairs = build_pairs(label_to_id)
    return SegInfo(id_to_label, pairs)


COLS = ["image_id",
        "r_ank_x", "r_ank_y", "r_ank_v",
        "r_kne_x", "r_kne_y", "r_kne_v",
        "r_hip_x", "r_hip_y", "r_hip_v",
        "l_hip_x", "l_hip_y", "l_hip_v",
        "l_kne_x", "l_kne_y", "l_kne_v",
        "l_ank_x", "l_ank_y", "l_ank_v",
        "b_pel_x", "b_pel_y", "b_pel_v",
        "b_spi_x", "b_spi_y", "b_spi_v",
        "b_nec_x", "b_nec_y", "b_nec_v",
        "b_hea_x", "b_hea_y", "b_hea_v",
        "r_wri_x", "r_wri_y", "r_wri_v",
        "r_elb_x", "r_elb_y", "r_elb_v",
        "r_sho_x", "r_sho_y", "r_sho_v",
        "l_sho_x", "l_sho_y", "l_sho_v",
        "l_elb_x", "l_elb_y", "l_elb_v",
        "l_wri_x", "l_wri_y", "l_wri_v"]


@cache_result_on_disk('cached/lip', [0, 1], forced=False)
def make_dataset(data_path, split="train"):
    """
    Makes the LIP dataset.
    TODO Test set will not work.
    """
    # load images
    logger = get_logger()

    if split == "train":
        img_data_path = os.path.join(data_path, 'train_images')
        seg_data_path = os.path.join(data_path, 'TrainVal_parsing_annotations', 'train_segmentations')
        pose_anno_path = os.path.join(data_path, 'TrainVal_pose_annotations', 'lip_train_set.csv')
    elif split == "val":
        img_data_path = os.path.join(data_path, 'val_images')
        seg_data_path = os.path.join(data_path, 'TrainVal_parsing_annotations', 'val_segmentations')
        pose_anno_path = os.path.join(data_path, 'TrainVal_pose_annotations', 'lip_val_set.csv')
    elif split == "test":
        # TODO
        img_data_path = os.path.join(data_path, 'test_images')
        seg_data_path = None
        pose_anno_path = None
        raise NotImplementedError

    pose_anno = pd.read_csv(pose_anno_path, header=0, names=COLS)
    joint_info = make_joint_info()
    data = []
    for index, datum in pose_anno.iterrows():
        image_id = datum['image_id'][:-len('.jpg')]
        img_path = os.path.join(img_data_path, image_id + '.jpg')
        if not os.path.isfile(img_path):
            logger.warning('File %s was not found', img_path)
            continue


        seg_path = os.path.join(seg_data_path, image_id + '.png')

        if not os.path.isfile(seg_path):
            logger.warning('File %s was not found', seg_path)
            continue

        coords = datum[1:]
        coords = coords.reshape(-1, 3)
        # drop visual column
        coords = coords[:, [0, 1]]
        head_size = None

        # TODO  Is this correct
        head_size = np.linalg.norm(coords[joint_info.ids.b_head] - coords[joint_info.ids.b_neck])
        d = {
                'path': img_path,
                'coords': coords,
                'seg_path': seg_path,
                'head_size': head_size
        }
        data.append(d)

    header = {
                'path': HeaderItem((), ""),
                'coords': HeaderItem((), ""),
                'seg': HeaderItem((), "")
             }
    seg_info = make_seg_info()
    info = {
        'joint_info': joint_info,
        'num_joints': joint_info.n_joints,
        'seg_info': seg_info,
        'num_seg_classes': len(CLASSES)
    }
    return data, header, info


@register_dataset('lip')
class Lip(PoseDataset):
    """
    Look into person
    """
    def __init__(self, data, header, info, flip_prob, *args, **kwargs):
        super().__init__("lip", data, header, info, *args, **kwargs)
        seg_info = info['seg_info']
        joint_info = info['joint_info']
        self.flip_prob = flip_prob
        self.flip_transform = FliplrWithPairs(p=flip_prob,
                keypoint_pairs=joint_info.mirror_mapping_pairs,
                segmentation_pairs=seg_info.pairs)

    def __getitem__(self, index):
        datum = self.data[index]
        datum = datum.copy()

        img = self.loader_fn(datum['path'])
        shape = img.shape
        coords = datum['coords']
        # image is a 3 channel png with identical channels
        seg = np.array(self.loader_fn(datum['seg_path']))[:, :, 0]

        if self.transform is not None:
            # flip transform is outside the pipeline
            # segmentation label flipping is not yet supported
            # do before possible normalization
            num_seg_classes = self.info['num_seg_classes']

            if self.flip_prob > 0:
                # only execute if the probability is greater 0
                # if the image will be flipped is decided by augmenter
                det_flip = self.flip_transform.to_deterministic()
                #det_flip = self.flip_transform
                img = det_flip.augment_image(img)
                seg = ia.SegmentationMapOnImage(seg, shape=seg.shape, nb_classes=num_seg_classes)
                seg = det_flip.augment_segmentation_maps(seg).get_arr_int()

                keypoints_on_image = ia.KeypointsOnImage.from_coords_array(coords, shape=shape)
                keypoints_on_image = det_flip.augment_keypoints([keypoints_on_image])
                coords = keypoints_on_image[0].get_coords_array()

            self.transform.to_deterministic()
            img = self.transform.augment_image(img)
            seg = self.transform.augment_segmentation(seg, num_seg_classes)
            # the shape of the original image
            coords = self.transform.augment_keypoint(coords, shape)
            # the shape of the augmented image
            coords = self.normalize_pose_keypoints(coords, img.shape)

        # we need to save the shape to restore the orginal coordinates
        datum['height'] = shape[0]
        datum['width'] = shape[1]
        datum['coords'] = coords
        datum['img'] = img
        # TODO why long?? Otherwise error in loss
        datum['seg'] = np.array(seg, dtype=np.int64)

        return datum

    def __len__(self):
        return len(self.data)

    @staticmethod
    def build(cfg, *args, **kwargs):
        split = cfg['split']
        evaluate = cfg.get('evaluate', 'both')
        #default to zero to avoid messing up validation
        flip_prob = cfg.get('flip_prob', 0.0)
        data_dir = Config.LIP_DATA
        data, header, info = make_dataset(data_dir, split)
        transform = transform_builder.build(cfg['transform'], info)
        dataset = Lip(data, header, info, flip_prob, transform, *args, **kwargs)
        # TODO very temporay solution
        # Looking for a better solution building the evaluation
        # to avoid passing too many parameters.
        dataset.evaluate_mode = evaluate
        return dataset

    def get_evaluation(self, model):
        pose = segmentation = False

        if self.evaluate_mode == 'pose':
            pose = True
        elif self.evaluate_mode == 'segmentation':
            segmentation = True
        else:
            pose = segmentation = True

        joint_info = self.info['joint_info']
        num_seg_classes = self.info['num_seg_classes']
        if pose and segmentation:
            print("LIP: Pose and Segmentation Evaluation started")
            return LipPoseSegmentationEvaluation(model, joint_info, num_seg_classes)
        elif pose:
            print("LIP: Pose Evaluation started")
            joint_info = self.info['joint_info']
            return LipPoseEvaluation(model, joint_info)
        elif segmentation:
            print("LIP: Segmentation Evaluation started")
            return LipSegmentationEvaluation(model, num_seg_classes)

        raise RuntimeError("Not the expected outputs available")


class LipSegmentationEvaluation(Evaluation):
    def __init__(self, model, num_classes):
        super().__init__("Lip")
        self.num_classes = num_classes
        self.hist = np.zeros((num_classes, num_classes))

    def get_writer(self, output_path):
        # for now do everything in memory
        self.writer = DummyWriter()
        return self.writer

    def before_saving(self, endpoints, data):
        # Change to Update and remove get_writer function?
        predictions = torch.argmax(endpoints['sem-logits'], dim=1).detach().cpu().numpy()
        # batch size of one
        assert predictions.shape[0] == 1

        pred = predictions[0]
        gt = data['seg'].detach().cpu().numpy()[0]
        self.hist += fast_hist(gt.flatten(), pred.flatten(), self.num_classes)

        return {}

    def score(self):
        score = calc_seg_score(self.hist)
        return score

class LipPoseEvaluation(Evaluation):
    def __init__(self, model, joint_info):
        super().__init__("Lip")
        self.joint_info = joint_info

    def get_writer(self, output_path):
        # for now do everything in memory
        self.writer = MemoryWriter()
        return self.writer

    def before_saving(self, endpoints, data):
        data_to_write = {
            "pose": endpoints['pose'].cpu(),
            "coords": data['coords'],
            "head_size": data['head_size'],
            "height": data['height'],
            "width": data['width']
        }
        return data_to_write

    @staticmethod
    def _score(pose, coords, height, width, head_size, ids):
        # no inplace
        pose = pose.copy()
        coords = coords.copy()

        # coords are between 0 and 1, rescale for correct error
        # broadcast to all joints

        pose[:, :, 0] *= width[:, None]
        pose[:, :, 1] *= height[:, None]

        coords[:, :, 0] *= width[:, None]
        coords[:, :, 1] *= height[:, None]
        def calc_dist(array1, array2):
            return np.linalg.norm(array1 - array2, axis=2)
        # TODO ignore head not visible in evaluation
        dist = calc_dist(pose, coords)
        pck_all, pck_joint = calculate_pckh(dist, head_size)

        score = {}
        sn = "PCKh {} @ {}"
        #threshold: values
        for t, v in pck_joint.items():
            score[sn.format(t, "Head")] = (v[ids['b_head']] + v[ids['b_neck']]) / 2
            score[sn.format(t, "Shoulder")] = (v[ids['l_sho']] + v[ids['r_sho']]) / 2
            score[sn.format(t, "Elbow")] = (v[ids['l_elb']] + v[ids['r_elb']]) / 2
            score[sn.format(t, "Wrist")] = (v[ids['l_wri']] + v[ids['r_wri']]) / 2
            score[sn.format(t, "Hip")] = (v[ids['l_hip']] + v[ids['r_hip']]) / 2
            score[sn.format(t, "Knee")] = (v[ids['l_kne']] + v[ids['r_kne']]) / 2
            score[sn.format(t, "Ankle")] = (v[ids['l_ank']] + v[ids['r_ank']]) / 2

        for t, v in pck_all.items():
            score[sn.format(t, "All")] = v
        return score

    def score(self):
        data = self.writer.data
        height = np.concatenate(data['height'])
        width = np.concatenate(data['width'])
        head_size = np.concatenate(data['head_size'])
        pose = np.concatenate(data['pose']) # prediction
        coords = np.concatenate(data['coords']) # gt
        return self._score(pose, coords, height, width, head_size, self.joint_info.ids)


class LipPoseSegmentationEvaluation(Evaluation):
    def __init__(self, model, joint_info, num_seg_classes):
        super().__init__("Lip")
        self.pose = LipPoseEvaluation(model, joint_info)
        self.seg = LipSegmentationEvaluation(model, num_seg_classes)

    def get_writer(self, output_path):
        self.writer = MemoryWriter()
        self.seg.writer = self.writer
        self.pose.writer = self.writer
        return self.writer

    def before_saving(self, endpoints, data):
        pose_data = self.pose.before_saving(endpoints, data)
        seg_data = self.seg.before_saving(endpoints, data)
        return {**pose_data, **seg_data}

    def score(self):
        pose_score = self.pose.score()
        seg_score = self.seg.score()
        return {**pose_score, **seg_score}
