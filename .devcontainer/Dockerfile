# Use Ubuntu 20.04 (Focal) for ROS Noetic
FROM ubuntu:20.04

# avoid any interactive prompts and ensure tput works
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=noetic \
    USERNAME=vscode \
    USER_UID=1000 \
    USER_GID=1000 \
    TERM=xterm

# 1) Install tools to add ROS apt repo and terminfo database
RUN apt-get update && apt-get install -y \
    curl \
    gnupg2 \
    lsb-release \
    ncurses-term \
    && rm -rf /var/lib/apt/lists/*

# 2) Add the ROS1 (Noetic) package repository
RUN curl -sSL http://packages.ros.org/ros.key | apt-key add - \
 && sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" \
           > /etc/apt/sources.list.d/ros1-latest.list'

# 3) Install ROS Noetic, Gazebo support, and common dev tools
RUN apt-get update && apt-get install -y \
    ros-noetic-ros-base \
    ros-noetic-gazebo-ros-pkgs \
    ros-noetic-gazebo-ros-control \
    sudo \
    git \
    python3-pip \
    python3-venv \
    build-essential \
    cmake \
    nano \
    vim \
    && rm -rf /var/lib/apt/lists/*

# 4) Create a non-root user for development
RUN groupadd --gid $USER_GID $USERNAME \
 && useradd -m --uid $USER_UID --gid $USER_GID -s /bin/bash $USERNAME \
 && usermod -aG sudo $USERNAME \
 && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" >> /etc/sudoers

WORKDIR /home/$USERNAME
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME

# switch to non-root user
USER $USERNAME

# 5) Ensure ROS & Gazebo are sourced in every new shell
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
 && echo "source /usr/share/gazebo/setup.sh"  >> ~/.bashrc

# 6) Clone the Interbotix ROS Rovers repo (LoCoBot support)
RUN git clone -b noetic https://github.com/Interbotix/interbotix_ros_rovers.git ros_rovers

# 7) Run the XSLocobot install script for Noetic + Create3 base
RUN chmod +x ros_rovers/interbotix_ros_xslocobots/install/amd64/xslocobot_amd64_install.sh \
 && ros_rovers/interbotix_ros_xslocobots/install/amd64/xslocobot_amd64_install.sh \
      -d $ROS_DISTRO -b create3 -n

# 8) Default entrypoint: source ROS and drop into an interactive bash
ENTRYPOINT ["/bin/bash", "-c", "source /opt/ros/$ROS_DISTRO/setup.bash && exec bash"]
