import os
import json
import random


# This is designed to create an abstraction for reading and writing 
# the candidates and their stats so I can move it to a database or 
# FTP server.
class IOLayer:
    def __init__(self, config_folder):
        self.config_folder = config_folder
        self.index = 6

    # Get the data shared by all configurations
    def fetch_all_config(self):
        with open(os.path.join(self.config_folder, "_all.json")) as f:
            result = json.load(f)
        return result

    # Get the configuration-specific data
    def fetch_config(self, config_name):
        with open(os.path.join(self.config_folder, f"{config_name}.conf")) as f:
            result = json.load(f)
        return result

    # Save the configuration
    def store_config(self,config):
        with open(os.path.join(self.config_folder, f"{config['name']}.conf"), "w") as f:
            json.dump(config, f, indent=2)

    # Create and return a unique name for a new candidate
    def new_name(self):
        v = random.randint(0, 1000000000)
        return f"{v:08X}"

    # The first step will be to train and test all the initial configs
    def claim_name_of_an_untested_configuration(self):
        # FIXME: This should return one untrained/tested config
        # It should only give this out once.
        # Returns None if there are no untested configs
        return "0"

    # The second step will be to mate the best configurations
    def names_of_best_configurations(self, n):
        # FIXME: This just returns the first n
        return [f"{i:08X}" for i in range(n)]
    
    # Store the results of a test
    def store_test_results(self, config_name, test_results):
        pass
