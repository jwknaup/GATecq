import json
import random

# This is designed to create an abstraction for reading and writing 
# the candidates and their stats so I can move it to a database or 
# FTP server.
class io_layer:
    def fetch_all_config(self):
        with open("../initial_configs/_all.json") as f:
            result = json.load(f)
        return result

    def fetch_config(self, config_name):
        with open(f"../initial_configs/{config_name}.conf") as f:
            result = json.load(f)
        return result

    def store_config(self,config):
        with open(f"../initial_configs/{config['name']}.conf", "w") as f:
            json.dump(config, f, indent=2)

    def new_name(self):
        v = random.randint(0, 1000000000)
        return f"{v:08X}"