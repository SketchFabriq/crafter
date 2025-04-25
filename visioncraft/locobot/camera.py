#!/usr/bin/env python3
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Float64
from cv_bridge import CvBridge
import cv2


class LocobotCamera:
    def __init__(self,
                 img_topic='/locobot/camera/color/image_raw',
                 pan_topic='/locobot/pan_controller/command',
                 tilt_topic='/locobot/tilt_controller/command'):

        if not rospy.core.is_initialized():
            rospy.init_node('locobot_camera_control', anonymous=True)

        self.bridge = CvBridge()
        self.image  = None
        rospy.Subscriber(img_topic, Image, self._img_cb)

        self._pan_pub  = rospy.Publisher(pan_topic,  Float64, queue_size=1)
        self._tilt_pub = rospy.Publisher(tilt_topic, Float64, queue_size=1)

        # give topics a moment to connect
        rospy.sleep(0.5)

    def _img_cb(self, msg):
        try:
            self.image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            rospy.logerr(f"cv_bridge: {e}")

    def get_image(self):
        return self.image

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
