#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from visioncraft.utils.transform_utils import transform_point_to_base_frame
from image_geometry import PinholeCameraModel


class LocobotCamera:
    def __init__(self,
                 img_topic='/locobot/camera/color/image_raw',
                 depth_topic='/locobot/camera/depth_registered/points',
                 depth_img_topic='/locobot/camera/aligned_depth_to_color/image_raw',
                 pan_topic='/locobot/pan_controller/command',
                 tilt_topic='/locobot/tilt_controller/command',
                 use_simulation=True):
        
        self.use_simulation = use_simulation

        if not rospy.core.is_initialized():
            rospy.init_node('locobot_camera_control', anonymous=True)

        self.bridge = CvBridge()
        self.image = None
        self.points = None
        self.depth_image = None
        self.cam_model = None

        if not self.use_simulation:
            self.cam_model = PinholeCameraModel()   
            info_msg = rospy.wait_for_message(
                '/locobot/camera/aligned_depth_to_color/camera_info',
                CameraInfo,
                timeout=2.0)
            self.cam_model.fromCameraInfo(info_msg)   


        rospy.Subscriber(img_topic, Image, self._img_cb)
        rospy.Subscriber(depth_img_topic, Image, self._depth_img_cb) # real     
        rospy.Subscriber(depth_topic, PointCloud2, self._depth_cb) # gazebo

        self._pan_pub = rospy.Publisher(pan_topic, Float64, queue_size=1)
        self._tilt_pub = rospy.Publisher(tilt_topic, Float64, queue_size=1)

        # give topics a moment to connect
        rospy.sleep(0.5)

    def _img_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'rgb8')
        except Exception as e:
            rospy.logerr(f"cv_bridge: {e}")

    def _depth_cb(self, msg):
        try:
            self.points = np.array(list(pc2.read_points(msg, skip_nans=True)))
        except Exception as e:
            rospy.logerr(f"point cloud conversion: {e}")

    def _depth_img_cb(self, msg):
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        self.depth_image = depth_img.astype(np.float32) * 0.001


    def get_point_at_pixel(self, x, y):
        """Get the 3D point at pixel coordinates (x,y)"""
        if self.use_simulation:
            if self.points is None:
                return None
            
            height, width = self.image.shape[:2]
            if x < 0 or x >= width or y < 0 or y >= height:
                return None
                
            idx = y * width + x
            if idx >= len(self.points):
                return None
                
            return self.points[idx][:3]
        else:
            return self.depth_to_xyz(x, y)

    def depth_to_xyz(self, u, v):
        """
        (u,v)     : pixel coordinate in colour image frame
        returns   : (x,y,z) in the **camera optical frame**
        """
        Z = float(self.depth_image[v, u])
        if Z == 0.0 or np.isnan(Z): 
            return None
        X = (u - self.cam_model.cx()) * Z / self.cam_model.fx()
        Y = (v - self.cam_model.cy()) * Z / self.cam_model.fy()
        return np.array([X, Y, Z], dtype=np.float32)
    
    def get_image(self):
        """Get the latest image from the camera"""
        if self.image is None:
            return None
        return self.image
    
    def get_depth_image(self):
        """Get the latest depth image from the camera"""
        if self.depth_image is None:
            return None
        return self.depth_image

    
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
        if self.use_simulation:
            point_base = transform_point_to_base_frame(point_camera,
                                                       source_frame='locobot/camera_depth_link',
                                                       target_frame='locobot/base_link')
        else:
            point_base = transform_point_to_base_frame(point_camera,
                                                       source_frame='locobot/camera_depth_optical_frame',
                                                       target_frame='locobot/base_link')
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
