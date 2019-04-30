from models import BaseModel, register_model
from builders import model_builder
from models.semantic_segmentation import FpnSemHead
from models.baseline import BaselineReidBranch


@register_model("reid_segmentation")
class ReidSeg(BaseModel):
    def create_endpoints(self):
        endpoints = {}
        endpoints.update(self.sem_head.create_endpoints())
        endpoints["emb"] = None
        endpoints["triplet"] = None
        return endpoints

    def __init__(self, backbone, num_classes):
        super().__init__()
        # expects fpn to return feature pyramid
        self.backbone = backbone
        # fpn has output dim 256
        self.sem_head = FpnSemHead(num_classes, 256)
        self.reid_head = BaselineReidBranch.build({"pooling": "max"})
        self.endpoints = self.create_endpoints()

    def forward(self, data, endpoints):
        # fpn also returns first layer before feature pyramid
        # orig_size
        _, _, h, w = data['img'].size()
        feature_pyramid, reid_x = self.backbone(data)
        if w == 256:
            endpoints, emb = self.sem_head(feature_pyramid, (h, w), endpoints)
        else:
            endpoints, reid_emb = self.reid_head(reid_x, endpoints)
            endpoints["emb"] = reid_emb
        return endpoints

    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        num_classes = cfg['num_seg_classes']
        model = ReidSeg(backbone, num_classes)
        skips = ["fc"]
        duplicates = []
        return model, skips, duplicates
