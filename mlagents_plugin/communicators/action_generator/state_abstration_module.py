import yaml

class StateAstractionModule:
    def __init__(self, settings_path: str):
        pass
        #with open(settings_path, 'r') as f:
        #    config = yaml.safe_load(f)

    def discretize(self, raw_state):
        return raw_state