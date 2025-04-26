#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image, PointCloud2
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2
import numpy as np
import sensor_msgs.point_cloud2 as pc2
from visioncraft.utils.transform_utils import transform_point_to_base_frame

class LocobotCamera:
    def __init__(self,
                 img_topic='/locobot/camera/color/image_raw',
                 depth_topic='/locobot/camera/depth_registered/points',
                 pan_topic='/locobot/pan_controller/command',
                 tilt_topic='/locobot/tilt_controller/command'):

        if not rospy.core.is_initialized():
            rospy.init_node('locobot_camera_control', anonymous=True)

        self.bridge = CvBridge()
        self.image = None
        self.points = None
        rospy.Subscriber(img_topic, Image, self._img_cb)
        rospy.Subscriber(depth_topic, PointCloud2, self._depth_cb)

        self._pan_pub = rospy.Publisher(pan_topic, Float64, queue_size=1)
        self._tilt_pub = rospy.Publisher(tilt_topic, Float64, queue_size=1)

        # give topics a moment to connect
        rospy.sleep(0.5)

    def _img_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge: {e}")

    def _depth_cb(self, msg):
        try:
            # Convert point cloud to numpy array
            self.points = np.array(list(pc2.read_points(msg, skip_nans=True)))
        except Exception as e:
            rospy.logerr(f"point cloud conversion: {e}")

    def get_image(self):
        return self.image

    def get_points(self):
        return self.points

    def get_point_at_pixel(self, x, y):
        """Get the 3D point at pixel coordinates (x,y)"""
        if self.points is None:
            return None
        
        # Find the closest point to the pixel coordinates
        # Assuming points are ordered in the same way as the image pixels
        height, width = self.image.shape[:2]
        if x < 0 or x >= width or y < 0 or y >= height:
            return None
            
        # Calculate the index in the point cloud array
        idx = y * width + x
        if idx >= len(self.points):
            return None
            
        return self.points[idx]

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
