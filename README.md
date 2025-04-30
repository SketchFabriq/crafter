# crafter


## Commands

### Starting the Robot using moveit

```bash
roslaunch interbotix_xslocobot_moveit_interface xslocobot_moveit_interface.launch robot_model:=locobot_wx250s show_lidar:=true use_gazebo:=true use_cpp_interface:=true
```

rosrun xacro xacro \
  $(rospack find interbotix_xslocobot_descriptions)/urdf/locobot.urdf.xacro \
  robot_model:=locobot_wx250s \
  arm_model:=mobile_wx250s \
  base_model:=create3 \
  robot_name:=locobot \
  show_lidar:=false \
  show_gripper_bar:=true \
  show_gripper_fingers:=true \
  external_urdf_loc:="" \
> ~/ros_ws/locobot_full.urdf


```bash
roslaunch locobot_connector cube_spawner.launch num_cubes:=5 pattern:=grid
```


export ROS_MASTER_URI=http://172.19.0.128:11311

docker-compose up -d --build


On robot : 
roslaunch interbotix_xslocobot_control xslocobot_control.launch robot_model:=locobot_wx250s use_base:=true

On host : 
roslaunch interbotix_xslocobot_moveit xslocobot_moveit.launch robot_model:=locobot_wx250s dof:=6 use_actual:=true


lst : 
roslaunch interbotix_xslocobot_moveit xslocobot_moveit.launch robot_model:=locobot_wx250s dof:=6 use_actual:=true use_camera:=true
