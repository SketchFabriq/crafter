#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from visioncraft.utils.transform_utils import transform_point_to_base_frame
from image_geometry import PinholeCameraModel                         # new


topic_map = {
    "real": {
        'img_topic': '/locobot/camera/color/image_raw',
        'depth_topic': '/locobot/camera/depth/image_rect_raw',
        'pan_topic': '/locobot/pan_controller/command',
        'tilt_topic': '/locobot/tilt_controller/command'
    },
    "sim": {
        'img_topic': '/locobot/camera/color/image_raw',
        'depth_topic': '/locobot/camera/depth_registered/points',
        'pan_topic': '/locobot/pan_controller/command',
        'tilt_topic': '/locobot/tilt_controller/command'
    }
}


class LocobotCamera:
    def __init__(self,
                 img_topic='/locobot/camera/color/image_raw',
                 depth_topic='/locobot/camera/depth/image_rect_raw',
                 pan_topic='/locobot/pan_controller/command',
                 tilt_topic='/locobot/tilt_controller/command'):

        if not rospy.core.is_initialized():
            rospy.init_node('locobot_camera_control', anonymous=True)

        self.bridge = CvBridge()
        self.image = None
        self.points = None
        self.cam_model = PinholeCameraModel()   

        rospy.Subscriber(img_topic, Image, self._img_cb)
        rospy.Subscriber(depth_topic, Image, self._depth_img_cb)
        info_msg = rospy.wait_for_message(
            '/locobot/camera/depth/camera_info',
            CameraInfo,
            timeout=2.0)
        self.cam_model.fromCameraInfo(info_msg)   
        
        self._pan_pub = rospy.Publisher(pan_topic, Float64, queue_size=1)
        self._tilt_pub = rospy.Publisher(tilt_topic, Float64, queue_size=1)

        # give topics a moment to connect
        rospy.sleep(0.5)

    def _img_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge: {e}")

    def _depth_img_cb(self, msg):
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        rospy.loginfo(f"encoding: {msg.encoding}")
        self.depth = depth_img

    def get_image(self):
        return self.image

    def get_points(self):
        return self.points

    def get_point_at_pixel(self, x, y):
        """Get the 3D point at pixel coordinates (x,y)"""
        return None if self.depth is None else self.depth[y, x]

    def depth_to_xyz(self, u, v):
        """
        (u,v)     : pixel coordinate in colour image frame
        returns   : (x,y,z) in the **camera optical frame**
        """
        Z = float(self.depth[v, u])         # metres
        if Z == 0.0 or np.isnan(Z):        # 0 == invalid for RealSense
            return None
        X = (u - self.cam_model.cx()) * Z / self.cam_model.fx()
        Y = (v - self.cam_model.cy()) * Z / self.cam_model.fy()
        return np.array([X, Y, Z], dtype=np.float32)

    
    def pan(self, angle_rad, wait_s=0.5):
        """Rotate head_pan_joint to angle_rad"""
        self._pan_pub.publish(Float64(angle_rad))
        rospy.sleep(wait_s)

    def tilt(self, angle_rad, wait_s=0.5):
        """Rotate head_tilt_joint to angle_rad"""
        self._tilt_pub.publish(Float64(angle_rad))
        rospy.sleep(wait_s)

    def look(self, pan_rad, tilt_rad, wait_s=0.5):
        """Move both axes at once."""
        self._pan_pub.publish(Float64(pan_rad))
        self._tilt_pub.publish(Float64(tilt_rad))
        rospy.sleep(wait_s)

    def get_target_coordinate_from_camera(self, point_camera):
        point_base = transform_point_to_base_frame(point_camera)
        return point_base

if __name__ == '__main__':
    cam = LocobotCamera()
    rate = rospy.Rate(1)
    while not rospy.is_shutdown():
        # sweep left/right while you watch the image
        for pan_angle in [ -1.0, 0.0, +1.0, 0.0 ]:
            cam.pan(pan_angle)
            img = cam.get_image()
            if img is not None:
                cv2.imshow("LoCoCam", img)
                cv2.waitKey(1)
        rate.sleep()
