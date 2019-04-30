from collections import defaultdict
from pycocotools.coco import COCO
import numpy as np

"""
from pycocotools.cocostuffeval import COCOStuffeval
class COCOPanopticEvalHelper(COCOStuffeval):

    def __init__(self, cocoGt, cocoRes, stuffStartId=1, stuffEndId=200):
        super().__init__(cocoGt, cocoRes, stuffStartId, stuffEndId, addOther=False)

    def summarize(self):
        '''
        Compute and display the metrics for leaf nodes only.
        :return: tuple of (general) stats and (per-class) statsClass
        '''

        # Check if evaluate was run and then compute performance metrics
        if not self.eval:
            raise Exception('Error: Please run evaluate() first!')

        # Compute confusion matrix for supercategories
        confusion = self.confusion

        # Compute performance
        [miou, fwiou, macc, pacc, ious, maccs] = self._computeMetrics(confusion)

        # Store metrics
        stats = np.zeros((4,))
        stats[0] = self._printSummary('Mean IOU', 'leaves', miou)
        stats[1] = self._printSummary('FW IOU', 'leaves', fwiou)
        stats[2] = self._printSummary('Mean accuracy', 'leaves', macc)
        stats[3] = self._printSummary('Pixel accuracy', 'leaves', pacc)

        # Store statsClass
        statsClass = {
            'ious': ious,
            'maccs': maccs
        }
        self.stats, self.statsClass = stats, statsClass

        return stats, statsClass

"""
class COCOPanopticHelper(COCO):
    # inspired from pycocoutils
    def __init__(self, annotation_file_path):

        self.ids = dict()

        super().__init__(annotation_file_path)


    def createIndex(self):
        print('creating index...')
        anns, cats, imgs = {}, {}, {}
        catToImgs = defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                anns[ann['image_id']] = ann

        if 'images' in self.dataset:
            for img in self.dataset['images']:
                imgs[img['id']] = img

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                for segm in ann['segments_info']:
                    catToImgs[segm['category_id']].append(ann['image_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.imgToAnns = None
        self.catToImgs = catToImgs
        self.imgs = imgs
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[], areaRng=[], iscrowd=None):
        raise ValueError("Only one annotation per image")


    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        raise NotImplementedError("Not supported anymore")


