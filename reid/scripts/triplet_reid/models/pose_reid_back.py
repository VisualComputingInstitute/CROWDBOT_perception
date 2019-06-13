import torch
import torch.nn as nn
from models.pose import SoftArgMax2d, Pose2DHead
from models import register_model, BaseModel
from builders import model_builder


@register_model("pose_reid_semi")
class PoseReidSemi(BaseModel):
    """
    Multi-branch architecture for PoseReid with a connection
    between pose and reid branch.
    """
    @property
    def dimensions(self):
        return {"emb": (self.dim,), "pose": (self.num_joints, 2)}

    def create_endpoints(self):
        endpoints = {}
        endpoints['emb'] = None
        endpoints['pose'] = None
        endpoints['pose_semi'] = None
        endpoints['r_pose'] = None
        return endpoints

    def __init__(self, backbone, num_joints, split):
        super().__init__()
        # backbone with two outputs
        self.backbone = backbone

        self.num_joints = num_joints
        self.p_pose_conv = nn.Conv2d(2048, num_joints, 1)
        self.pooling = nn.AdaptiveMaxPool2d(1)

        self.split = split
        self.softargmax = SoftArgMax2d()
        # is the reid head split or not
        if self.split:
            self.dim = 2048 - self.num_joints
            self._forward_fn = self._forward_split
        else:
            self.dim = 2048
            self.r_pose_conv = nn.Conv2d(2048, num_joints, 1)
            self._forward_fn = self._forward_no_split

        self.endpoints = self.create_endpoints()

    def forward(self, x, endpoints):
        return self._forward_fn(x, endpoints)

    def _forward_no_split(self, x, endpoints):
        p, r = self.backbone(x)

        p_heatmap = self.p_pose_conv(p)
        #print(pose_data.shape)
        endpoints['pose'] = self.softargmax(p_heatmap)

        if p.shape[-2:] == (16, 16):
            # quick and dirty skip pose images
            endpoints['triplet'] = None
            endpoints['emb'] = None
            endpoints['pose_semi'] = None
            endpoints['r_pose'] = None
            return endpoints

        # no gradient between the two sides
        p_pose = endpoints['pose'].detach()

        r_pose = self.r_pose_conv(r)
        r_pose = self.softargmax(r_pose)

        triplet_data = self.pooling(r)
        triplet_data = triplet_data.view(triplet_data.size(0), -1)
        endpoints['triplet'] = [triplet_data]
        endpoints['emb'] = triplet_data
        endpoints['pose_semi'] = p_pose
        endpoints['r_pose'] = r_pose
        return endpoints

    def _forward_split(self, x, endpoints):
        p, r = self.backbone(x)

        p_heatmap = self.p_pose_conv(p)
        #print(pose_data.shape)
        endpoints['pose'] = self.softargmax(p_heatmap)

        if p.shape[-2:] == (16, 16):
            # quick and dirty skip pose images
            endpoints['triplet'] = None
            endpoints['emb'] = None
            endpoints['pose_semi'] = None
            endpoints['r_pose'] = None
            return endpoints

        # no gradient between the two sides
        pose_predictions = endpoints['pose'].detach()

        r, pose_info_r = torch.split(r, [self.dim, self.num_joints], dim=1)

        r_pose = self.softargmax(pose_info_r)

        triplet_data = self.pooling(r)
        triplet_data = triplet_data.view(triplet_data.size(0), -1)
        endpoints['triplet'] = [triplet_data]
        endpoints['emb'] = triplet_data
        endpoints['pose_semi'] = pose_predictions
        endpoints['r_pose'] = r_pose
        return endpoints


    @staticmethod
    def build(cfg):
        backbone = model_builder.build(cfg['backbone'])
        split = cfg['split']
        model = PoseReidSemi(backbone, cfg['num_joints'], split)
        return model, [], []

