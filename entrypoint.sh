#!/bin/bash
source /$HOME/jackal_ws/devel/setup.bash
cd /$HOME/jackal_ws/src/gatecq
#chmod +x /jackal_ws/src/gatecq/src/planner_instance.py
#ls /venv/bin/
#which python3
#export ROS_LOG_DIR=/jackal_ws/
#roslaunch-logs
exec ${@:1}
