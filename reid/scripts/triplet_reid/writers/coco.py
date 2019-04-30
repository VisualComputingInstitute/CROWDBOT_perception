import json
import os
import numpy as np
from PIL import Image
import settings
import writers.coco_utils as cu
from logger import get_tensorboard_logger


class COCOJsonWriter(object):
    def __init__(self, output_filename):
        self.output_filename = output_filename
        self.first = True

    def write(self, predictions, ids, inverse, inverse_labels=None):
        from pycocotools.cocostuffhelper import segmentationToCocoResult

        for i in range(predictions.shape[0]):

            prediction = predictions[i, :]
            img_id = ids[i]

            prediction = np.squeeze(prediction)

            if inverse:
                prediction = inverse_labels[prediction]

            anns = segmentationToCocoResult(prediction, img_id, stuffStartId=inverse_labels[0])
            for ann in anns:
                ann['segmentation']['counts'] = ann['segmentation']['counts'].decode('utf-8')

            str_ = json.dumps(anns)
            # ignore []
            str_ = str_[1:-1]

            output_string = ""
            if self.first:
                self.first = False
            else:
                output_string = ","

            if len(str_.strip()) > 0:
                output_string += str_
            else:
                output_string = ""
                print("No valid output for imgId: %d" % img_id)

            self.file.write(output_string)

    def __enter__(self):
        self.file = open(self.output_filename, "w")
        self.file.write("[")
        # not sure if this good style
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.file.write("]")
        self.file.close()
        self.file = None
        # clean up
        # only if all values are set
        if (exc_type and exc_value and tb):
            os.remove(self.output_filename)


def panopticSegmentationToCocoResult(sem_seg, inst_ids, img_id, file_name, cats):
    '''
    Convert a panoptic segmentation to coco result format
    :param sem_seg: semantic segmentation map
    :param inst_ids: instance ids map
    :param img_id: the image_id
    :param file_name: the corresponding file_name
    :return: a dictionary that describes the json object and an inst image
    '''

    generator = cu.IdGenerator(cats)
    h, w = inst_ids.shape
    nice_instance_map = np.zeros((3, h, w))
    segments = []
    identifiers, indices = np.unique(inst_ids, return_index=True)
    for identifier, index in zip(identifiers, indices):
        new_id, color = generator.get_id_and_color(int(sem_seg.flat[index]))
        for channel, val in enumerate(color):
            nice_instance_map[channel][inst_ids == identifier] = val
        segment = {
            "id": int(new_id),
            "category_id": int(sem_seg.flat[index])
        }
        segments.append(segment)

    res = {
        "image_id": int(img_id),
        "file_name": str(file_name),
        "segments_info": segments
    }
    return res, nice_instance_map


class COCOPanopticJSONWriter(object):
    def __init__(self, output_filename):
        self.output_filename = output_filename
        self.output_dir = output_filename.split(".json")[0]
        if not os.path.isdir(self.output_dir):
            os.mkdir(self.output_dir)
        self.first = True
        self.tensorboard_logger = get_tensorboard_logger()

        with open(settings.Config.COCO_PANOPTIC_CATEGORIES) as fp:
            categories = json.load(fp)
            self.categories = {entry["id"]: entry for entry in categories}

    def write(self, predicted_sem, predicted_ids, ids, file_names, inverse, inverse_labels=None):

        for i in range(predicted_sem.shape[0]):

            sem_seg = predicted_sem[i, :]
            inst_ids = predicted_ids[i, :]
            img_id = ids[i]
            file_name = file_names[i]

            sem_seg = np.squeeze(sem_seg)

            if inverse:
                sem_seg = inverse_labels[sem_seg]

            anns, pretty_instance_map = panopticSegmentationToCocoResult(sem_seg, inst_ids, img_id, file_name, self.categories)

            str_ = json.dumps(anns)
            # ignore []
            str_ = str_[1:-1]
            str_ = "{" + str_ + "}"

            output_string = ""
            if self.first:
                self.first = False
            else:
                output_string = ","

            if len(str_.strip()) > 0:
                output_string += str_
            else:
                output_string = ""
                print("No valid output for imgId: %d" % img_id)

            self.file.write(output_string)
            image_filename = os.path.join(self.output_dir, file_name)
            im = Image.fromarray(pretty_instance_map.transpose((1, 2, 0)).astype(np.uint8))
            im.save(image_filename)
            self.tensorboard_logger.add_image("inst-pred", pretty_instance_map, dataformat="CHW")

    def __enter__(self):
        self.file = open(self.output_filename, "w")
        self.file.write("{")
        self.file.write("\"annotations\":")
        self.file.write("[")
        # not sure if this good style
        return self

    def __exit__(self, exc_type, exc_value, tb):
        self.file.write("]")
        self.file.write("}")
        self.file.close()
        self.file = None
        # clean up
        # only if all values are set
        if (exc_type and exc_value and tb):
            os.remove(self.output_filename)
