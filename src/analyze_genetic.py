import os
import numpy as np
import json
import matplotlib.pyplot as plt
import iolayer


if __name__ == '__main__':
    num_generations = 9
    num_instances = 4
    config_folder = '../initial_configs/'
    io_layer = iolayer.IOLayer(config_folder)
    fig, ((ax2, ax3), (ax4, ax5)) = plt.subplots(2, 2)
    for ii in range(num_generations):
        generation_configs = [num_instances*ii, num_instances*ii+1, num_instances*ii+2, num_instances*ii+3]
        generation_scores = []
        generation_lrs = []
        generation_eps = []
        generation_discount = []
        generation_depth = []
        generation = [ii, ii, ii, ii]
        for jj in generation_configs:
            conf = io_layer.fetch_config(jj)
            res = conf['total_reward_test']
            lr = conf['learning_rate']
            eps = conf['epsilon']
            discount = conf['discount_factor']
            depth = len(conf['layers'])
            generation_scores.append(res)
            generation_lrs.append(lr)
            generation_eps.append(eps)
            generation_discount.append(discount)
            generation_depth.append(depth)
        # ax1.plot(generation, generation_scores, '.')
        ax2.plot(generation, generation_lrs, 'o')
        ax3.plot(generation, generation_eps, 'o')
        ax4.plot(generation, generation_discount, 'o')
        ax5.plot(generation, generation_depth, 'o')
        print(generation_depth)
    ax2.set_title('Learning Rate')
    ax3.set_title('Epsilon')
    ax4.set_title('Discount Factor')
    ax5.set_title('Depth')
    ax2.set_xlabel('generation')
    ax3.set_xlabel('generation')
    ax4.set_xlabel('generation')
    ax5.set_xlabel('generation')
    print(np.mean(generation_lrs), np.mean(generation_eps), np.mean(generation_discount), np.mean(generation_depth))
    # plt.tight_layout()
    plt.show()
