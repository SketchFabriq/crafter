import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2

class LocobotCamera:
    def __init__(self, topic='/locobot/camera/color/image_raw'):
        # Initialize ROS node (anonymous so you can run multiple)
        rospy.init_node('locobot_camera_node', anonymous=True)
        self.bridge = CvBridge()

        # Storage for the latest OpenCV image
        self.image = None

        # Subscribe to camera topic
        self.sub = rospy.Subscriber(topic, Image, self._image_callback)

        rospy.sleep(1.0)
        rospy.loginfo(f"Subscribed to {topic}, waiting for images...")

    def _image_callback(self, msg):
        """Callback: convert ROS Image to CV2 and store."""
        try:
            cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
            self.image = cv_img
        except CvBridgeError as e:
            rospy.logerr(f"cv_bridge error: {e}")

    def get_image(self):
        """Return the most recent CV2 image (or None if none received yet)."""
        if self.image is None:
            rospy.logwarn_throttle(5, "No image received yet")
        return self.image

    def save_image(self, filename):
        """Write the latest image to disk."""
        if self.image is not None:
            cv2.imwrite(filename, self.image)
            rospy.loginfo(f"Image saved to {filename}")
        else:
            rospy.logwarn("Cannot save image: none received")

if __name__ == '__main__':
    cam = LocobotCamera()

