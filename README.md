# gatecq
Our entry into the BARN competition

## Setup 
1. Create a ROS workspace in your $HOME directory
2. Follow the instructions at 
https://github.com/Daffan/nav-competition-icra2022
to set up the nav-competition repo and its dependencies inside your `jackwal_ws/src/` folder
3. Clone this repo to the `src/` folder
4. `catkin_make` the workspace
5. Install the python requirements

## Running Locally
A single instance of the planner can be trained locally
by running `local_run.sh`.
This file will source the required ROS environment files
and launch the sim and RL nodes

Note: this script assumes `$HOME/jackal_ws`.
Additionally, you may need to modify the shebang line at the top of 
planner_instance.py to point to your python executable for this project
which you can find by running `which python` (or `which python3` etc depending on how you run python).
Sorry I couldn't find a good way to make this more general.

## Running batches in Singularity
1. Ensure you have installed Singularity and its dependencies by following the instructions at
https://github.com/Daffan/nav-competition-icra2022
2. Check to make sure the python version in the shebang line at the top of planner_instance.py matches
the python version installed in the Singularity image in the Singularityfile.def file
3. Build the singularity image `sudo singularity build --notest planner_sim_instance_image.sif Singularityfile.def`.
This will take some time but should only need to be done once unless you need to change the installed packages
4. Run `batch_run.sh`. This will run the spawn_planners.py script which will run multiple singularity images simultaneously,
where each contains its own ROS network, simulation, and `planner_instance`. The environments will train and then `spawn_planners`
will breed new configs and launch new planners for multiple generations.
