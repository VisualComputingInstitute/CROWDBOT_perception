#!/usr/bin/env python
# Ros ReID verify node

import rospy
from sensor_msgs.msg import Image as RosImage
from cv_bridge import CvBridge, CvBridgeError
from message_filters import ApproximateTimeSynchronizer, Subscriber
from frame_msgs.msg import DetectedPersons
import cv2
import numpy as np
from numpy.linalg import norm


class ReIdVerify:

    def __init__(self, image_topic, embeded_persons_topic, queue_size):
        # count is for debug and verify
        self.count = 0;

        # init self member
        self._prev_image = None  # just initialize something
        self._prev_bbx = None
        self._bridge = CvBridge()
        self._image_sub = Subscriber(image_topic, RosImage)
        self._bboxs_sub = Subscriber(embeded_persons_topic, DetectedPersons)
        self._approxi_sync = ApproximateTimeSynchronizer([self._image_sub, self._bboxs_sub], queue_size=queue_size, slop=0.5) # what is the slop???
        self._approxi_sync.registerCallback(self.callback)

    def callback(self, ros_image, ros_bboxs):
        if len(ros_bboxs.detections) == 0: # if the boundingbox is empty, just do nothing and go to publish
            pass
        else:
            try:
                cv_image = self._bridge.imgmsg_to_cv2(ros_image, "rgb8")
                cur_image = cv_image.copy()
            except CvBridgeError as e:
                print(e)
            # get prev image and boundingbox
            if self._prev_bbx is None: # if its the first time run, do nothing just get the image and the first bounding box
                self._prev_image = cv_image.copy()
                self._prev_bbx = ros_bboxs.detections[0] # get the first bounding box
                return

            # we want to get a len 3 det_list
            # the first is person 1 in current frame
            # the second in person 2 in current frame
            # the third is person 1 in prev frame
            det_list = list()
            det_list.append(ros_bboxs.detections[0])
            if len(ros_bboxs.detections) == 1: # if only see one person, we simply copy it
                det_list.append(ros_bboxs.detections[0])
            else:
                det_list.append(ros_bboxs.detections[1])

            det_list.append(self._prev_bbx)

            # compute three dets's similarity, L2 norm
            sim_list = list()
            for det1 in det_list:
                for det2 in det_list:
                    emb1 = np.array(det1.embed_vector)
                    emb2 = np.array(det2.embed_vector)
                    sim_list.append(norm(emb1-emb2))



            # draw the first bounding box in the current frame
            font = cv2.FONT_HERSHEY_SIMPLEX
            fontScale = 1
            fontColor = (255, 255, 255)
            lineType = 2
            textLocationbias = 30;

            sim_ind = ['1-1','1-2','1-3','2-1','2-2','2-3','3-1','3-2','3-3']
            cv2.namedWindow("current_frame", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
            cv2.namedWindow("prev_frame", cv2.WINDOW_NORMAL)  # Create window with freedom of dimensions
            for i in range(2):
                det = det_list[i]
                x_min = int(det.bbox_x)
                y_min = int(det.bbox_y)
                w = int(det.bbox_w)
                h = int(det.bbox_h)
                cv2.rectangle(cv_image, (x_min, y_min), (x_min+w, y_min+h), (255, 0, 0), 10)
                # draw the index of this bbx

                cv2.putText(cv_image,str(i+1),
                            (x_min+30,y_min+30),
                            font,
                            fontScale,
                            fontColor,
                            lineType)

                cv2.putText(cv_image,sim_ind[3*i+0]+":"+format(sim_list[3*i+0],'.1f'),
                            (x_min+w/2,y_min+h/2),
                            font,
                            fontScale,
                            fontColor,
                            lineType)

                cv2.putText(cv_image, sim_ind[3*i+1]+":"+format(sim_list[3*i+1],'.1f'),
                            (x_min+w/2,y_min+h/2+textLocationbias),
                            font,
                            fontScale,
                            fontColor,
                            lineType)
                cv2.putText(cv_image, sim_ind[3*i+2]+":"+format(sim_list[3*i+2],'.1f'),
                            (x_min + w / 2, y_min + h / 2 + 2*textLocationbias),
                            font,
                            fontScale,
                            fontColor,
                            lineType)

            cv2.imshow("current_frame",cv_image)
            cv2.waitKey(5)
            # draw previous frame
            det = det_list[2]
            x_min = int(det.bbox_x)
            y_min = int(det.bbox_y)
            w = int(det.bbox_w)
            h = int(det.bbox_h)
            cv2.rectangle(self._prev_image, (x_min, y_min), (x_min + w, y_min + h), (255, 0, 0), 10)

            cv2.putText(self._prev_image, str(3),
                        (x_min + 30, y_min + 30),
                        font,
                        fontScale,
                        fontColor,
                        lineType)

            cv2.putText(self._prev_image, sim_ind[6] + ":" + format(sim_list[6], '.1f'),
                        (x_min + w / 2, y_min + h / 2 ),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.putText(self._prev_image, sim_ind[7] + ":" + format(sim_list[7], '.1f'),
                        (x_min + w / 2, y_min + h / 2 + 1 * textLocationbias),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.putText(self._prev_image, sim_ind[8] + ":" + format(sim_list[8], '.1f'),
                        (x_min + w / 2, y_min + h / 2 + 2 * textLocationbias),
                        font,
                        fontScale,
                        fontColor,
                        lineType)
            cv2.imshow("prev_frame", self._prev_image)
            cv2.waitKey(5)

            # update the prev frame
            self._prev_image = cur_image.copy()
            self._prev_bbx = ros_bboxs.detections[0]


            # for idx, det in enumerate(ros_bboxs.detections):
            #     det.embed_vector = emb_batch[idx].tolist()




if __name__ == '__main__':
    rospy.init_node('reid_verify', anonymous=True)
    image_topic = rospy.get_param('~image', default='/hardware/video/valeo/rectificationNIKRLeft/PanoramaImage')
    detected_persons_topic = rospy.get_param('~detected_persons', default='with_embed_detectedpersons')
    queue_size = rospy.get_param('~queue_size',default='10')

    reid_verify = ReIdVerify(image_topic,detected_persons_topic,queue_size)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        print("shutting down this reid node")