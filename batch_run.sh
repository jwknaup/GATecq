#!/bin/bash
source /opt/ros/$ROS_DISTRO/setup.bash
source $HOME/jackal_ws/devel/setup.bash
log=log_file.log
python3 src/spawn_planners.py > $log
