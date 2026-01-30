import yaml

class LLMSettings:

    def __init__(self, settings_path: str, other_settings: dict):
        with open(settings_path, 'r') as f:
            config = yaml.safe_load(f)

        llm_config = config.get('llm_settings', {})
        self.game_desc = llm_config.get('game_desc', '')
        self.agent_role = llm_config.get('agent_role', '')
        self.history_length = llm_config.get('history_length', 1)
        self.model = llm_config.get('model_name', '')
        self.task = llm_config.get('task', '')
        # self.model_constructor = MODEL_CONSTRUCTORS[self.model_name]

        raw_actions = llm_config.get('actions', {})
        
        self.actions = {
            "discrete": {},
            "continuous": {}
        }

        if 'discrete' in raw_actions:
            for item in raw_actions['discrete']:
                name = item['name']
                self.actions["discrete"][name] = {
                    "options": item.get('options', []),
                    "description": item.get('description', '')
                }

        if 'continuous' in raw_actions:
            for item in raw_actions['continuous']:
                name = item['name']
                self.actions["continuous"][name] = {
                    "options": item.get('options', []),
                    "description": item.get('description', ''),
                    "values": item.get('values', [0.0]*len(item.get('options', [])))
                }

        self.use_vectorial_obs = llm_config.get('use_vectorial_obs', False)
        self.use_visual_obs = llm_config.get('use_visual_obs', False)
        self.use_raycast_obs = llm_config.get('use_raycast_obs', False)
        self.use_grid_obs = llm_config.get('use_grid_obs', False)

        if self.use_grid_obs:
            self.grid_color_legend = llm_config.get('grid_color_legend', None)

        self.batch_size = llm_config.get('batch_size', None)
        
        self.num_agents = other_settings['num_agents']
        self.discrete_branches = other_settings['discrete_branches']
        self.num_continuous_actions = other_settings['num_continuous_actions']

        self.features = llm_config.get('features', False)


    def get_index_of_action(self, action: str, choice: str, continuous: bool) -> int:
        return self.actions['continuous' if continuous else 'discrete'][action]['options'].index(choice)
    
    def get_continuous_value_by_index(self, action, idx) -> float:
        return self.actions['continuous'][action]['values'][idx]
