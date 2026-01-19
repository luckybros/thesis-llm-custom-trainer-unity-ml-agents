import yaml
import numpy as np

class StateAstractionModule:
    def __init__(self, settings: dict):
        self.features = settings.features

        # dizionario dispatch in cui mappo il tipo di dato con la funzione
        self.processors = {
            'ONE_HOT': self._process_one_hot
            #'SCALAR': self._process_scalar,
            #'BOOLEAN': self._process_boolean
        }
        #with open(settings_path, 'r') as f:
        #    config = yaml.safe_load(f)

    def discretize(self, raw_state):
        """
        Input: 
            {'vectorial': 
                {'agent_0': [0.0, 1.0], 
                'agent_1': [1.0, 0.0], 
                'agent_2': [1.0, 0.0]}, 
        It should became
            {'vectorial': 
                {'agent_0': {
                                'abstract_state_0': 'abstract_state_value_0',
                                'abstract_state_1': 'abstract_state_value_1',
                                'abstract_state_2': 'abstract_state_value_2',
                            }
                'agent_1': {
                                'abstract_state_0': 'abstract_state_value_0',
                                'abstract_state_1': 'abstract_state_value_1',
                                'abstract_state_2': 'abstract_state_value_2',
                            }, 
                'agent_2': {
                                'abstract_state_0': 'abstract_state_value_0',
                                'abstract_state_1': 'abstract_state_value_1',
                                'abstract_state_2': 'abstract_state_value_2',
                            }
                }, 
        """
        #print(f"raw_state : {raw_state}")
        raw_state_vectorial = raw_state['vectorial']

        for agent_id, obs_vect in raw_state_vectorial.items():
            # per ogni agente prendo il vettore delle osservazioni
            # leggo le features
            # per ogni features leggo il tipo di osservazioni
            # chiamo la funzione relativa al tipo di osservazioze con il sottovettore con gli indici in cui si trova l'osservazione
            # trasformo il valore nel dizionario relativo all'agente in un altro dizionario
            # con {'nome_feature': 'valore'} per ogni feature
            np_obs_vect = np.array(obs_vect)

            abstract_state_dict = {}

            for feature_name, features_options in self.features.items():
                """
                current_objective: {'type': 'ONE_HOT', 
                                    'indices': [0, 1], 
                                    'mapping': 
                                        {0: 'FIND_GREEN_PLUS', 
                                        1: 'FIND_RED_X'}
                                    }
                """

                f_type = features_options['type']

                indices = features_options['indices']

                mapping_dict = features_options['mapping']

                sub_vector = np_obs_vect[indices]

                process_function = self.processors[f_type]

                abstract_value = process_function(sub_vector, mapping_dict)

                abstract_state_dict[feature_name] = abstract_value
            
            raw_state['vectorial'][agent_id] = abstract_state_dict

        #print(f"state after processing: {raw_state}")

        return raw_state
        

    def _process_one_hot(self, values: list, mapping: dict):
        """
        Find the value in the vector that contains one and return 
        the value corresponding in the mapping
        """
        idx = np.argmax(values)
        return mapping[idx]