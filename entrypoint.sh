source ~/robogym_ws/devel/setup.bash
cd /home/vscode/ros_ws && catkin_make
source /home/vscode/ros_ws/devel/setup.bash
echo \"source ~/ros_ws/devel/setup.bash\" >> /home/vscode/.bashrc
pip install -e .