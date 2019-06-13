import torch.nn as nn
import torch
from models.panoptic.sem_head import SemNet
from models import register_model
from models import BaseModel

@register_model("resnet-50-fpn-pan-net")
class PanNet(BaseModel):

    def __init__(self, num_classes, hard_attention=True):
        """
        :param num_classes:
        :param hard_attention:
        """
        super(PanNet, self).__init__()

        self.sem_head = SemNet(num_classes)
        self.endpoints = self.create_endpoints()
        self.use_hard_attention = hard_attention

        self.first_stuff_index = 80


    def forward(self, data, endpoints):
        """
        :param data: contains imgs: tensor of shape BxCxHxW and bboxs: tensor of shape BxNx5, includes N tuples of (ID, x, y, w, h)
        :param endpoints: the endpoints tensor
        :return: semantic and panoptic logits
        """

        if self.use_hard_attention:
            return self.forward_hard(data, endpoints)
        else:
            return self.forward_soft(data, endpoints)

    def forward_hard(self, data, endpoints):
        device = next(self.parameters()).device
        bboxs = data['gt-bboxs'].to(device, non_blocking=True)
        imgs = data['img'].to(device, non_blocking=True)

        endpoints = self.sem_head.forward(data, endpoints)

        num_batches = bboxs.size()[0]

        sem_logits = endpoints['sem-logits']
        # copy all stuff logits
        stuff_logits = sem_logits[:, self.first_stuff_index:, :, :]
        stuff_logits_size = stuff_logits.size()
        inst_logits = torch.zeros(num_batches, bboxs.size()[1], stuff_logits_size[2], stuff_logits_size[3]).to(imgs)

        # extract classes
        class_list_inst = bboxs[:, :, 0]
        class_list_stuff = torch.empty(num_batches, stuff_logits.size()[1]).to(imgs)
        for i in range(num_batches):
            class_list_stuff[i, :] = torch.arange(stuff_logits.size()[1]) + self.first_stuff_index

        class_list = torch.cat((class_list_stuff, class_list_inst), dim=1)

        # extract bbox coords
        bbox_tensor = bboxs[:, :, 1:]

        for i in range(bbox_tensor.size()[1]):
            if bbox_tensor[:, i, :].sum() == 0:
                break
            logit_slice_indices = class_list_inst[:, i].to(dtype=torch.long)
            inst_logits[:, i, :, :] = self._extract_bbox_padded(sem_logits[range(num_batches), logit_slice_indices, ...], bbox_tensor[:, i, :])

        pan_logits = torch.cat((stuff_logits, inst_logits), dim=1)
        max_indices = torch.argmax(pan_logits, dim=1)
        endpoints['inst-pred'] = max_indices
        endpoints['sem-pred'] = torch.empty_like(max_indices)
        for i in range(num_batches):
            endpoints['sem-pred'][i, ...] = class_list[i][max_indices[i]]

        return endpoints

    def _extract_bbox_padded(self, imgs, bboxes):
        imgs = torch.unsqueeze(imgs, dim=1)
        bboxes = bboxes.transpose(0, 1)

        # top left corner
        x0s = bboxes[0]
        y0s = bboxes[1]
        ws = bboxes[2]
        hs = bboxes[3]
        H = imgs.size()[2]
        W = imgs.size()[3]

        offsetx = 2*(x0s+(ws-1)/2)/W-1
        offsety = 2*(y0s+(hs-1)/2)/H-1
        scalex = ws/W
        scaley = hs/H
        zeros = torch.zeros(imgs.size()[0]).to(imgs)

        """
        ------------------------------------------------------------------
        |  bbox_width  |              |     (x0 + (bbox_width -1)/2      |
        |  ----------  |       0      | 2x  -----------------------  -1  |
        |  img_width   |              |             img_width            |
        |----------------------------------------------------------------|
        |              |  bbox_height |     (y0 + (bbox_height -1)/2     |
        |       0      |  ----------- | 2x  -----------------------  -1  |
        |              |  img_height  |             img_height           |
        |----------------------------------------------------------------|
        """

        theta = torch.stack((
                torch.stack((scalex, zeros, offsetx), dim=1),
                torch.stack((zeros, scaley, offsety), dim=1)),
                dim=1
                )
        #theta = torch.Tensor([[[w/W, 0, offsetx], [0, h/H, offsety]]])


        indices = nn.functional.affine_grid(theta, torch.Size((imgs.size()[0], 1, 256, 256)))

        output_up = nn.functional.grid_sample(imgs, indices, mode='bilinear', padding_mode='zeros')

        """
        -------------------------------------------------------------------------
        |   img_width  |              |     (img_width/2 - bbox_width/2 - x0)   |
        |  ----------  |       0      | 2x  --------------------------------    |
        |  bbox_width  |              |                 bbox_width              |
        |-----------------------------------------------------------------------|
        |              |   img_height |     (img_height/2 - bbox_height/2 - y0) |
        |       0      |  ----------- | 2x  ----------------------------------  |
        |              |  bbox_height |                 bbox_height             |
        |-----------------------------------------------------------------------|
        """

        offsetx = (W/2 - ws/2-x0s)/ws*2
        offsety = (H/2 - hs/2-y0s)/hs*2
        scalex = W/ws
        scaley = H/hs

        theta = torch.stack((
                torch.stack((scalex, zeros, offsetx), dim=1),
                torch.stack((zeros, scaley, offsety), dim=1)),
                dim=1
                )

        #theta = torch.Tensor([[[W/w, 0, offsetx], [0, H/h, offsety]]])

        indices = nn.functional.affine_grid(theta, (imgs.size()[0], 1, H, W))

        output = nn.functional.grid_sample(output_up, indices, mode='nearest', padding_mode='zeros')
        output = torch.squeeze(output, dim=1)
        return output

    def forward_soft(self, *x):
        pass

    @staticmethod
    def build(cfg):
        model = PanNet(cfg['num_classes'])
        skips = ["fc"]
        duplicates = []
        return model, skips, duplicates

    @staticmethod
    def create_endpoints():
        endpoints = {}
        endpoints['sem-pred'] = None
        endpoints['inst-pred'] = None
        endpoints.update(SemNet.create_endpoints())
        return endpoints