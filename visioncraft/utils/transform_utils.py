#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped

def transform_point_to_base_frame(point_camera, source_frame='locobot/camera_depth_link', target_frame='locobot/base_link'):
    """
    Transform a point from camera frame to robot base frame
    """
    tf_buffer = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)

    rospy.sleep(1.0)  # wait for camera and TF data

    point_stamped = PointStamped()
    point_stamped.header.frame_id = source_frame
    point_stamped.header.stamp = rospy.Time(0)
    point_stamped.point.x = point_camera[0]
    point_stamped.point.y = point_camera[1]
    point_stamped.point.z = point_camera[2]
    
    try:
        transform = tf_buffer.lookup_transform(target_frame, source_frame, rospy.Time(0), rospy.Duration(1.0))
        transformed_point = tf2_geometry_msgs.do_transform_point(point_stamped, transform)
        return [transformed_point.point.x, transformed_point.point.y, transformed_point.point.z]
    except Exception as e:
        rospy.logerr(f"TF transform failed: {e}")
        return None
