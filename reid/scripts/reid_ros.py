#!/usr/bin/env python
# Ros ReID node


from __future__ import print_function

# ros related package
import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from frame_msgs.msg import DetectedPersons

# follow is ReId package
import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.realpath(__file__))+'/triplet_reid') # tis dirname(realpath(__file__)) return the path of the folder which include this excuteable script
from triplet_reid.builders import model_builder
from torchvision import transforms
from PIL import Image



# For preprocessing
H = 256
W = 128
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

# Test time augmentation would give +1.5%

transform = transforms.Compose([
    transforms.Resize((H, W)),
    transforms.ToTensor(),
    transforms.Normalize(mean, std)
])
# Preprocessing end



class ReId:

    def __init__(self, image_topic, detected_persons_topic, queue_size, model_path, output_topic):
        # # count is for debug and verify
        # self.count = 0;

        from models.baseline import Baseline, BaselineReidBranch
        from models.backbone.resnet import GroupnormBackbone
        from resnet_groupnorm.resnet import Bottleneck
        backbone = GroupnormBackbone(Bottleneck, [3, 4, 6, 3], 16, 1)
        reid_branch = BaselineReidBranch.build({"pooling": "max"})
        self.model = Baseline(backbone, reid_branch)
        file_model_dic = model_builder.get_model_dic_from_file(model_path)
        file_model_dic = model_builder.clean_dict(file_model_dic)
        self.model.load_state_dict(file_model_dic)
        self.model = self.model.cuda()

        self._bridge = CvBridge()
        self._bboxs_pub = rospy.Publisher(output_topic, DetectedPersons, queue_size = 10)
        self._image_sub = Subscriber(image_topic, RosImage)
        self._bboxs_sub = Subscriber(detected_persons_topic, DetectedPersons)
        self._approxi_sync = ApproximateTimeSynchronizer([self._image_sub, self._bboxs_sub], queue_size=queue_size, slop=0.5) # what is the slop???
        self._approxi_sync.registerCallback(self.callback)

    @staticmethod
    def get_input_tensor_list( image_list):
        tensor_list = list()
        for img in image_list:
            tensor = transform(img).unsqueeze(0)  # add batch dim
            tensor = tensor.cuda()
            tensor_list.append(tensor)
        return tensor_list

    @staticmethod
    def cal_similarity( emb0, emb1):
        return torch.dist(emb0, emb1, 2)

    @staticmethod
    def batch_tensor( tensor_list):
        return torch.cat(tensor_list, 0)

    def callback(self, ros_image, ros_bboxs):
        if len(ros_bboxs.detections) == 0: # if the boundingbox is empty, just do nothing and go to publish
            pass
        else:
            try:
                cv_image = self._bridge.imgmsg_to_cv2(ros_image, "rgb8")
            except CvBridgeError as e:
                print(e)

            # read bounding boxes, take subimage to a list of image.
            person_image_list = list()
            for det in ros_bboxs.detections:
                x_min = int(det.bbox_x)
                y_min = int(det.bbox_y)
                w = int(det.bbox_w)
                h = int(det.bbox_h)
                x_max = x_min+w
                y_max = y_min+h
                person_image_list.append(cv_image[y_min:y_max, x_min:x_max])

            # convert cv2 image list to pil image list
            pil_image_list = list()
            for sub_img in person_image_list:
                pil_image_list.append(Image.fromarray(sub_img))

            tensor_list = self.get_input_tensor_list(pil_image_list)
            batched_tensor = self.batch_tensor(tensor_list)
            endpoints = self.model.infere({'img': batched_tensor})
            emb_batch = endpoints['emb']

            # insert the embed vector to each detectedPerson message
            for idx, det in enumerate(ros_bboxs.detections):
                det.embed_vector = emb_batch[idx].tolist()

        # finally we publish the result.
        self._bboxs_pub.publish(ros_bboxs)





if __name__ == '__main__':
    rospy.init_node('reid_node', anonymous=True)
    image_topic = rospy.get_param('~image', default='/hardware/video/valeo/rectificationNIKRLeft/PanoramaImage')
    detected_persons_topic = rospy.get_param('~detected_persons', default='/yoloconvertor_pano/detected_persons_left')
    queue_size = rospy.get_param('~queue_size',default='10')
    model_path = rospy.get_param('~model_path')
    output_topic = rospy.get_param('~embed_detectedpersons')
    reid = ReId(image_topic,detected_persons_topic,queue_size,model_path,output_topic)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down this reid node")

