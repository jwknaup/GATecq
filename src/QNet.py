import torch.nn as nn
import torch.nn.functional as F
import torch
import math

def activation_for_string(act_str):
    if act_str == "relu":
        return nn.ReLU()
    if act_str == "sigmoid":
        return nn.Sigmoid()
    if act_str == "tanh":
        return nn.Tanh()

class QNet(nn.Module):
    def __init__(self, config, all_config):
        super().__init__()
        self.other_inputs = all_config["other_inputs"]
        self.lidar_inputs = all_config["lidar_inputs"]
        # For hidden data for LSTM and GRU layers
        # Key is index of layer, value is the hidden state
        self.hidden_data = {}

        # Are doing convolution to reduce dimension of lidar data?
        if "lidar_conv" in config:
            lidar_dict = config["lidar_conv"]
            output_channels = lidar_dict["output_channels"]
            kernel_size = lidar_dict["kernel_size"]
            stride = lidar_dict["stride"]
            self.lidar_conv = nn.Conv1d(1, output_channels, kernel_size, stride=stride)
            len_out = math.floor((self.lidar_inputs - kernel_size - 2) / stride) + 1
            outshape = output_channels * len_out + self.other_inputs
            self.lidar_activation = activation_for_string(lidar_dict["activation"])

        else:
            outshape = self.other_inputs + self.lidar_inputs
            self.lidar_conv = None

        layer_dicts = config["layers"]
        self.layers = []
        for layer_dict in layer_dicts:
            layertype = layer_dict["type"]
            output_dim = layer_dict["output_dim"]
            if layertype == "linear":
                self.layers.append(nn.Linear(outshape, output_dim))
                outshape = output_dim
                self.layers.append(activation_for_string(layer_dict["activation"]))
                if "dropout" in layer_dict:
                    self.layers.append(nn.Dropout(layer_dict["dropout"]))
            elif layertype == "lstm":
                dropout = layer_dict["dropout"]
                self.layers.append(nn.LSTM(outshape, output_dim))
                if "dropout" in layer_dict:
                    self.layers.append(nn.Dropout(layer_dict["dropout"]))
                outshape = output_dim
            elif layertype == "gru":
                self.layers.append(nn.GRU(outshape, output_dim))
                if "dropout" in layer_dict:
                    self.layers.append(nn.Dropout(layer_dict["dropout"]))
                outshape = output_dim
            else:
                print(f"Unknown layer type: {layertype}")

        # The last layer is always fully-connected with linear activation
        self.layers.append(nn.Linear(outshape, all_config["possible_actions"]))

    # Reset the hidden state of the network between episodes
    def reset_hidden(self):
        self.hidden_data = {}
                                                       
    def forward(self, x):
        # Do we need to do convolution on the lidar data?
        if self.lidar_conv is not None:
            lidar_in = x[:, self.other_inputs:]
            other_in = x[:, :self.other_inputs]
            lidar_out = self.lidar_conv(lidar_in)
            lidar_out = self.lidar_activation(lidar_out)
            if len(lidar_out.shape) < 3:
                lidar_out = lidar_out.unsqueeze(0)
            # Add the other data back in
            lidar_out = torch.flatten(lidar_out, start_dim=1)
            x = torch.cat((other_in, lidar_out), dim=1)
        for i in range(len(self.layers)):
            # Do have hidden state data for this layer?
            if i in self.hidden_data:
                hidden_data = self.hidden_data[i]
                x, hidden_data = self.layers[i](x, hidden_data)
                # Update hidden state
                self.hidden_data[i] = hidden_data
            else:
                x = self.layers[i](x)

                # Is this layer an LSTM or GRU?
                if type(x) is tuple:
                    x, hidden_data = x
                    self.hidden_data[i] = hidden_data
            
        return x    





