#!/bin/bash
source /opt/ros/$ROS_DISTRO/setup.bash
source $HOME/jackal_ws/devel/setup.bash
roslaunch gatecq rl_sim_instance.launch gui:=true rl_config:=38
