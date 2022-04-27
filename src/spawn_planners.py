import subprocess
import numpy as np
import iolayer
import mate
import mutate
import correct


def main():
    num_generations = 2
    num_instances = 4
    config_folder = 'initial_configs'
    io_layer = iolayer.IOLayer(config_folder)
    # identify initial configs
    configs = [0, 1, 2, 3]
    # for num generations
    for ii in range(num_generations):
        # for # simultaneous instances:
        children = set()
        print('Running trainers with configs ', configs)
        for jj in range(num_instances):
            # spawn trainers
            cmd = ['./singularity_run.sh', 'planner_sim_instance_image.sif', 'roslaunch', 'gatecq', 'rl_trainer_sim_instance.launch', 'rl_config:={}'.format(configs[jj]), 'config_folder:={}'.format(config_folder)]
            p = subprocess.Popen(cmd)
            children.add(p)
        # join()
        for p in children:
            res = p.wait()
            if res == 0:
                print('rl_trainer_sim_instance finished successfully')
            else:
                print('ERROR: rl_trainer_sim_instance finished with ERROR!')
        # read results
        results = []
        for jj in range(num_instances):
            conf = io_layer.fetch_config(configs[jj])
            results.append(conf['total_reward_test'])
        # breed new generation
        new_configs = []
        weights = np.asarray(results) + abs(np.min(results)) + 1e-8
        weights = weights / np.sum(weights)
        print('Breeding ', configs, ' with probabilities ', weights)
        for jj in range(num_instances):
            parents = np.random.choice(configs, size=2, replace=False, p=weights)
            new_configs.append(io_layer.index)
            child = mate.mate(io_layer.fetch_config(parents[0]), io_layer.fetch_config(parents[1]), io_layer.index)
            io_layer.index += 1
            all_conf = io_layer.fetch_all_config()
            mutate.mutate_candidate(child, all_conf, 1.0, 1.0)
            # Basic sanity check/corrections for mutated candidates
            correct.correct_candidate(child, all_conf, 1.0)
            io_layer.store_config(child)
        configs = new_configs.copy()


if __name__ == '__main__':
    main()
