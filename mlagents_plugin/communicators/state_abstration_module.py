import yaml

class StateAstractionModule:
    def __init__(self, settings_path: str):
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)

        