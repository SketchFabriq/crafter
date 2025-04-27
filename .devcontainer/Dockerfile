FROM ubuntu:20.04

# noninteractive APT
ENV DEBIAN_FRONTEND=noninteractive \
    ROS_DISTRO=noetic \
    USERNAME=vscode \
    USER_UID=1000 \
    USER_GID=1000

# Install core tools
RUN apt-get update && apt-get install -y \
    curl git lsb-release sudo gnupg2 wget \
    python3-pip python3-venv build-essential cmake gdb nano vim \
    software-properties-common \
  && rm -rf /var/lib/apt/lists/*

# Add ROS repository and install Noetic
RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -cs) main" \
      > /etc/apt/sources.list.d/ros-latest.list \
  && curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc \
      | apt-key add - \
  && apt-get update && apt-get install -y \
    ros-noetic-desktop-full \
    python3-rosdep python3-rosinstall python3-rosinstall-generator python3-wstool python3-catkin-tools \
  && rm -rf /var/lib/apt/lists/*

# Initialize rosdep
RUN rosdep init && rosdep update

# Create non-root user for VS Code
RUN groupadd --gid $USER_GID $USERNAME \
  && useradd -m --uid $USER_UID --gid $USER_GID -s /bin/bash $USERNAME \
  && usermod -aG sudo $USERNAME \
  && echo "$USERNAME ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/$USERNAME

# Stub systemctl so Netplan/service calls no-op
RUN ln -sf /bin/true /usr/bin/systemctl

# Switch to non-root for workspace setup
USER $USERNAME
WORKDIR /home/$USERNAME

# Pre install requirements
COPY --chown=$USERNAME:$USERNAME requirements.txt /home/$USERNAME/
RUN python3 -m pip install --user -r requirements.txt

# Create Interbotix workspace
# RUN mkdir -p interbotix_ws/src

# Download & run LoCoBot installer (Noetic + Create3)
RUN curl -sSL https://raw.githubusercontent.com/Interbotix/interbotix_ros_rovers/main/interbotix_ros_xslocobots/install/amd64/xslocobot_amd64_install.sh -o xslocobot_install.sh \
    && chmod +x xslocobot_install.sh

ENV TERM=xterm
RUN ./xslocobot_install.sh -d noetic -b create3 -p ~/interbotix_ws -n

# Fix permissions
USER root
RUN chown -R $USERNAME:$USERNAME /home/$USERNAME
USER $USERNAME

# Auto-source ROS and workspace
RUN echo "source /opt/ros/$ROS_DISTRO/setup.bash" >> ~/.bashrc \
    && echo "source /home/$USERNAME/interbotix_ws/devel/setup.bash" >> ~/.bashrc


ENV ROBOGYM_WS=/home/$USERNAME/robogym_ws

# Install robogym server and build it
WORKDIR $ROBOGYM_WS/src
RUN git clone -b add_locobot https://github.com/montrealrobotics/robo-gym-robot-servers.git
RUN rm $ROBOGYM_WS/src/robo-gym-robot-servers/interbotix_rover_robot_server/CATKIN_IGNORE

WORKDIR $ROBOGYM_WS
RUN /bin/bash -c "source /home/$USERNAME/interbotix_ws/devel/setup.bash && catkin init && catkin build"
RUN echo "source $ROBOGYM_WS/devel/setup.bash" >> ~/.bashrc

# Clone and install robo-gym client
WORKDIR /home/$USERNAME
RUN git clone -b add_locobot https://github.com/montrealrobotics/robo-gym.git \
    && python3 -m pip install --user -e robo-gym

# Final working directory
WORKDIR /home/$USERNAME