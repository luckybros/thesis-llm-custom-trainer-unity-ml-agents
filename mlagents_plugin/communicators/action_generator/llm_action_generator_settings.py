import yaml

class LLMActionGeneratorSettings:

    def __init__(self, settings_path: str):
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)
        config = config.get('llm_settings')
        self.game_desc = config.get('game_desc', '')
        self.agent_role = config.get('agent_role', '')
        self.history_length = config.get('history_length', 1)
        self.model_name = config.get('model_name', '')
        self.task = config.get('task', '')
        self.actions = config.get('actions', [])
