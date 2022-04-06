import datetime
import numpy as np
import random
import logging
import QNet

# Basic sanity check/corrections for mutated candidates
def correct_candidate(candidate, all):

    # Overall, the dimension should go down as we pass through the network
    if 'lidar_conv' in candidate:
        conv = candidate['lidar_conv']
        current_dim = all['other_inputs'] + QNet.dim_for_convolution(all['lidar_inputs'], conv['kernel_size'], conv['stride'], conv['output_channels'])
    else:
        current_dim = all['other_inputs'] + all['lidar_inputs']

    for i in range(len(candidate['layers'])):
        layer = candidate['layers'][i]
        next_dim = layer['output_dim']

        # Should not be less than possible actions
        if next_dim < all['possible_actions']:
            if expansion > 1:   # If we're expanding, we should not shrink
                next_dim = random.randrange(all['possible_actions'], current_dim)
            else:
                next_dim = all['possible_actions']

            logging.debug(f"Expanded output of layer {i} to {next_dim}")
            layer['output_dim'] = next_dim

        # Should not be more than the previous layer
        if next_dim > current_dim:
            next_dim = current_dim
            logging.debug(f"Slimmed output of layer {i} to {next_dim}")
            layer['output_dim'] = next_dim

        current_dim = next_dim