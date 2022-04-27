import subprocess
import asyncio
import iolayer


def main():
    num_generations = 2
    num_instances = 2
    config_folder = 'initial_configs'
    io_layer = iolayer.IOLayer(config_folder)
    # identify initial configs
    configs = [0, 1]
    # for num generations
    for ii in range(num_generations):
        # for # simultaneous instances:
        children = set()
        for jj in range(num_instances):
            # spawn trainers
            cmd = ['./singularity_run.sh', 'planner_sim_instance_image.sif', 'roslaunch', 'gatecq', 'rl_trainer_sim_instance.launch', 'rl_config:={}'.format(configs[jj]), 'config_folder:={}'.format(config_folder)]
            p = subprocess.Popen(cmd)
            children.add(p)
        # join()
        for p in children:
            res = p.wait()
            print(res)
        # read results
        results = []
        for jj in range(num_instances):
            conf = io_layer.fetch_config(configs[jj])
            results.append(conf)
        # breed new generation
        # TODO breed next generation, write new configs, and update configs list


if __name__ == '__main__':
    main()
