import yaml
import numpy as np

class StateAstractionModule:
    def __init__(self, settings: dict):
        self.features = settings.features

        # dizionario dispatch in cui mappo il tipo di dato con la funzione
        self.processors = {
            'ONE_HOT': self._process_one_hot,
            'BUCKET': self._process_bucket,
            'BOOL': self._process_boolean
        }

        if "raycast" in self.features:
            self.num_detectable_tags = self.features['raycast']['objects_nearby']['num_detectable_tags']
            self.RAYCAST_OBS_OFFSET = self.num_detectable_tags + 2
            self.RAYCAST_OBS_CHECK_IDX = self.num_detectable_tags
            self.RAYCAST_OBS_DIST_IDX = self.num_detectable_tags + 1
        #with open(settings_path, 'r') as f:
        #    config = yaml.safe_load(f)

    def discretize(self, raw_state):
        if 'vectorial' in self.features:
            self.discretize_vectorial(raw_state=raw_state)
        if 'raycast' in self.features:
            self.discretize_raycast(raw_state=raw_state)
        return raw_state
    
    def discretize_vectorial(self, raw_state):
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
        raw_state_vectorial = raw_state['VECTORIAL']

        for agent_id, obs_vect in raw_state_vectorial.items():
            # per ogni agente prendo il vettore delle osservazioni
            # leggo le features
            # per ogni features leggo il tipo di osservazioni
            # chiamo la funzione relativa al tipo di osservazioze con il sottovettore con gli indici in cui si trova l'osservazione
            # trasformo il valore nel dizionario relativo all'agente in un altro dizionario
            # con {'nome_feature': 'valore'} per ogni feature
            np_obs_vect = np.array(obs_vect)

            abstract_state_dict = {}

            for feature_name, features_options in self.features['vectorial'].items():
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
            
            raw_state['VECTORIAL'][agent_id] = abstract_state_dict

        #print(f"state after processing: {raw_state}")

        return raw_state
    
    def discretize_raycast(self, raw_state):

        raw_state_raycast = raw_state['RAYCAST'] # PER ORA SOLO UNO

        for agent_id, ray_vect in raw_state_raycast.items():
            # itero sugli agenti
            """
            {'agent_0': 
                [0.0, 1.0, 0.0, 0.0, 0.5465336441993713, 
                 0.0, 1.0, 0.0, 0.0, 0.6310825943946838, 
                 0.0, 1.0, 0.0, 0.0, 0.6310826539993286, 
                 0.0, 0.0, 1.0, 0.0, 0.670921266078949, 
                 0.0, 0.0, 0.0, 1.0, 1.0, 
                 0.0, 0.0, 1.0, 0.0, 0.5810348391532898, 
                 0.0, 0.0, 0.0, 1.0, 1.0]}
            }
            """

            # itero ogni 5 valori
            abstract_state_dict = {}

            for i in range(0, len(ray_vect), self.RAYCAST_OBS_OFFSET):
                # quarto posto è quello che ci dice se ha colpito qualcuno, se è 1 non lo consideriamo proprio e andiamo al prossimo
                # itero sui raggi 
                ray_subvect = ray_vect[i:i+self.RAYCAST_OBS_OFFSET]
                if ray_subvect[self.RAYCAST_OBS_CHECK_IDX] == 1:
                    continue

                for feature_name, feature_options in self.features['raycast'].items():

                    sub_ray_vect = ray_subvect[0:feature_options['num_detectable_tags']]
                    detected_obj = self._process_one_hot(sub_ray_vect, feature_options['detectable_tags'])
                    direction = feature_options['direction'][i//self.RAYCAST_OBS_OFFSET]
                    distance = ray_subvect[self.RAYCAST_OBS_DIST_IDX]
                    buck_distance = self._process_bucket(distance, feature_options['distance'])

                    abstract_state_dict.setdefault(feature_name, []).append({
                        'OBJECT': detected_obj, 
                        'DIRECTION': direction, 
                        'DISTANCE': buck_distance
                    })
                
            raw_state['RAYCAST'][agent_id] = abstract_state_dict

        return raw_state


    def _process_one_hot(self, values: list, mapping: dict):
        """
        Find the value in the vector that contains one and return 
        the value corresponding in the mapping
        """
        idx = np.argmax(values)
        return mapping[idx]
    
    def _process_boolean(self, value, mapping):
        return mapping[int(value)]
    
    def _process_bucket(self, value, mapping):
        for bucket_options in mapping:
            if bucket_options['min'] <= value <= bucket_options['max']:
                return bucket_options['label']
        

    
