#!/usr/bin/env python3
import rospy
import tf2_ros
import tf2_geometry_msgs
from geometry_msgs.msg import PointStamped
import numpy as np

def transform_point_to_base_frame(point_optical,
                                  source_frame='locobot/camera_depth_optical_frame',
                                  target_frame='locobot/base_link'):
    tf_buffer   = tf2_ros.Buffer()
    tf_listener = tf2_ros.TransformListener(tf_buffer)
    rospy.sleep(0.2)

    ps = PointStamped()
    ps.header.frame_id = source_frame
    ps.header.stamp    = rospy.Time.now()
    ps.point.x, ps.point.y, ps.point.z = point_optical

    tfm  = tf_buffer.lookup_transform(target_frame,
                                       source_frame,
                                       rospy.Time(0),
                                       rospy.Duration(1.0))
    out  = tf2_geometry_msgs.do_transform_point(ps, tfm)
    return [out.point.x, out.point.y, out.point.z]

