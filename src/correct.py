import datetime
import numpy as np
import random
import logging
import QNet

# Basic sanity check/corrections for mutated candidates
def correct_candidate(candidate, all):

    # Every candidate should have at least one recurrent layer
    has_recurrent = False
    layers = candidate['layers']
    for layer in layers:
        if layer['type'] == 'lstm' or layer['type'] == 'gru':
            has_recurrent = True
            break

    if not has_recurrent:
        location = len(layers)//2
        if random.random() < 0.5:
            type = 'lstm'
        else:
            type = 'gru'
            
        if location < len(layers) - 1:
            # What's the output dim of what's already there?
            output_dim = layers[location]['output_dim']
        else:
            output_dim = all['possible_actions']

        new_layer = {'type': type, 'output_dim':output_dim, 'dropout': 0.1}
        logging.debug(f"No memory: Adding {new_layer} at {location}")
        candidate['layers'].insert(location, new_layer)

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
            if expansion > 1:   
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