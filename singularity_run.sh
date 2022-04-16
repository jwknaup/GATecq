#!/bin/bash
singularity exec -i --nv -n --network=none -p ${1} /bin/bash /jackal_ws/src/gatecq/entrypoint.sh ${@:2}
