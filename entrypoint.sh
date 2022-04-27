#!/bin/bash
source /$HOME/jackal_ws/devel/setup.bash
cd /$HOME/jackal_ws/src/gatecq
exec ${@:1}
