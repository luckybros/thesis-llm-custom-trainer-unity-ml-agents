import os
import numpy as np
import re

class PromptBuilder:

    _DIR_PROSE = {
        "FAR_LEFT": "far left", "LEFT": "left", "SLIGHT_LEFT": "slightly left",
        "FORWARD": "directly ahead", "SLIGHT_RIGHT": "slightly right",
        "RIGHT": "right", "FAR_RIGHT": "far right", "CENTER": "center",
    }
    _DIST_PROSE = {
        "IMMEDIATE": "immediate", "CLOSE": "close", "MEDIUM": "medium", "FAR": "far",
    }
    _OBJ_PROSE = {
        "TANK": "enemy tank", "POWER UP": "power-up",
        "WALL": "wall", "OBSTACLE": "obstacle",
    }
 
    _DIR_SHORT  = {"LEFT": "L", "SLIGHT_LEFT": "SL", "FORWARD": "FW",
                   "SLIGHT_RIGHT": "SR", "RIGHT": "R", "FAR_RIGHT": "FR",
                   "FAR_LEFT": "FL", "CENTER": "C"}
    _DIST_SHORT = {"IMMEDIATE": "I", "CLOSE": "C", "MEDIUM": "M", "FAR": "F"}
    _OBJ_SHORT  = {"OBSTACLE": "OBS", "TANK": "TANK", "POWER UP": "PWUP", "WALL": "WALL"}

    def __init__(self, settings):
        
        self.game_overview = settings.game_desc
        self.agent_role = settings.agent_role
        self.task = settings.task
        self.actions = settings.actions
        self.agent_histories = {}
        self.max_history = 3
        self.actions = settings.actions

        self.obs_template = settings.obs_template
        self.raycast_keys = settings.raycast_keys

        self.use_vectorial = settings.use_vectorial_obs
        self.use_visual = settings.use_visual_obs
        self.use_raycast = settings.use_raycast_obs
        self.use_grid = settings.use_grid_obs
        if self.use_grid:
            self.grid_color_legend = settings.grid_color_legend


        self._initialize_system_message()

    def build_prompt(self, agent_id: str, state: dict):
        """Return {'sys_msg': str, 'hum_msg': list} ready for the LLM."""
        content_text = self._build_obs_text(agent_id, state)

        human_text = (
            f"### HISTORY\n{self._format_history(agent_id)}\n\n"
            f"### CURRENT OBSERVATIONS\n{content_text}"
        )

        content = [{"type": "text", "text": human_text}]
       
        for key in ["VISUAL", "GRID"]:
            if (key == "VISUAL" and self.use_visual) or (key == "GRID" and self.use_grid):
                images = state.get(key, {})
                for agent_id, img_b64 in images.items():
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                    })

        return {"sys_msg": self.system_msg, "hum_msg": content}
    
    def _build_obs_text(self, agent_id: str, state: dict) -> str:
        """
        Render the observation for *agent_id* as a string.
        Used both inside build_prompt and externally (e.g. to store history).
        """
        vec = state.get("VECTORIAL", {}).get(agent_id, {})
        raycasts = {
            key: state.get(key, {}).get(agent_id, [])
            for key in self.raycast_keys
        }

        if self.obs_template:
            return self._render_template(vec, raycasts) 
        return None
    
    def update_history(self, agent_id: str, state: dict, action_taken: dict):
        """
        Da chiamare ad ogni step DOPO aver eseguito l'azione.
        action_taken es: {'Move': 'Move forward', 'Turn': 'Turn right', 'Shoot': "Don't shoot"}
        """
        if agent_id not in self.agent_histories:
            self.agent_histories[agent_id] = []

        obs_text = self._build_obs_text(agent_id, state)
        discrete_action = action_taken[0]['discrete']
        action_str = ", ".join([f"{k}: {v}" for k, v in discrete_action.items()])
        entry = f"Obs: {obs_text.strip()} → {action_str}"

        self.agent_histories[agent_id].append(entry)

        if len(self.agent_histories[agent_id]) > self.max_history:
            self.agent_histories[agent_id].pop(0)

    def get_action_from_response_list(self, action_list):

        result = {0: {'discrete': {}}}

        discrete_settings = self.actions['discrete']

        for index, (action_name, action) in enumerate(discrete_settings.items()):
            action_options = action['options']

            print(f"action list: {action_list}")
            chosen_index = action_list[index]

            chosen_options = action_options[chosen_index]
            result[0]['discrete'][action_name] = chosen_options
        
        return result
    
    def get_action_from_policy_list(self, action_list):
        # {'discrete': {'agent_0-0': [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]]}}

        result = {0: {'discrete': {}}}

        action_list = action_list['discrete']['agent_0-0']
        # [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 0.0]]

        discrete_settings = self.actions['discrete']

        for index, (action_name, action_options) in enumerate(discrete_settings.items()):
            # i:0, k:Move, v:{'options': ['Stay', 'Move forward', 'Move backward'], 'description': 'Move the character forward or backward.\n'}
            action_from_index = action_list[index]
            idx_of_chosen_action = np.argmax(action_from_index)
            chosen_option = action_options['options'][idx_of_chosen_action]
            result[0]['discrete'][action_name] = chosen_option

        return result

    def _format_history(self, agent_id: str):
        history = self.agent_histories.get(agent_id, [])
        if not history:
            return "None"
        lines = []
        for i, entry in enumerate(history):
            step = i - len(history)  # es. -4, -3, -2, -1
            lines.append(f"Step {step}: {entry}")
        return "\n".join(lines)
    
    def _render_template(self, vec, raycasts):
        """Fill obs_template with discretized values and prose raycast sentences."""
        text = self.obs_template

        for key, value in vec.items():
            text = text.replace(f"{{{key}}}", str(value))

        for ray_key, ray_data in raycasts.items():
            # Derive a human-readable side label from the key name:
            # RAYCAST_FRONT → "front", RAYCAST_BACK → "back", RAYCAST_LEFT → "left", …
            side = ray_key.replace("RAYCAST_", "").lower()
            text = text.replace(f"{{{ray_key}}}", self._raycast_to_prose(ray_data, side))

        text = re.sub(r"\{[^}]+\}", "?", text)

        return text.strip()
    
    def _raycast_to_prose(self, raycast_data: list, side: str) -> str:
        """
        Convert a raycast entry list into a natural-language sentence.
 
        Example:
          "On its front: enemy tank slightly right at medium range, wall right at close range."
        """
        detections = []
        for entry in raycast_data:
            dir_label  = self._DIR_PROSE.get(entry["direction"],  entry["direction"].lower())
            for obj in entry.get("objects_detected", []):
                obj_label  = self._OBJ_PROSE.get(obj["object"],   obj["object"].lower())
                dist_label = self._DIST_PROSE.get(obj["distance"], obj["distance"].lower())
                detections.append(f"{obj_label} {dir_label} at {dist_label} range")
 
        if not detections:
            return f"On its {side}: nothing detected."
        return f"On its {side}: {', '.join(detections)}."

    def _initialize_system_message(self):    
        #abbreviation_legend = (
        #    "## DATA ABBREVIATIONS (Mandatory for efficiency)\n"
        #    "- RF / RB: Raycast Front / Raycast Back\n"
        #    "- Directions: L (Left), SL (Slight Left), FW (Forward), SR (Slight Right), R (Right), FR (Far Right), C (Center)\n"
        #    "- Objects: OBS (Obstacle), TANK (Enemy), PWUP (PowerUp)\n"
        #    "- Distances: IM (Immediate), CL (Close), MD (Medium), FA (Far)\n"  
        #)

        self.system_msg = (
            f"### [GAME OVERVIEW]\n{self.game_overview}\n\n"
            f"### [ROLE & TASK]\n{self.agent_role}\nTask: {self.task}\n\n"
        #    f"{abbreviation_legend}\n"
            f"### [ACTION SET]\n{self._format_action_schema()}\n\n"
        )

        if self.use_grid:
            self.system_msg += f"\n### [GRID LEGEND]\n{self._format_grid_color_schema()}"

    def _compress_raycast(self, raycast_data):
        """Trasforma il JSON pesante in una stringa tecnica ultra-densa."""
        d_map = {'IMMEDIATE': 'I', 'CLOSE': 'C', 'MEDIUM': 'M', 'FAR': 'F'}
        o_map = {'OBSTACLE': 'OBS', 'TANK': 'TANK', 'POWER UP': 'PWUP'}
        dir_map = {'LEFT': 'L', 'SLIGHT_LEFT': 'SL', 'FORWARD': 'FW',  
                   'SLIGHT_RIGHT': 'SR', 'RIGHT': 'R', 'FAR_RIGHT': 'FR', 'CENTER': 'C'}

        if not isinstance(raycast_data, list): return str(raycast_data)

        compressed_parts = []
        for entry in raycast_data:
            direction = dir_map.get(entry['direction'], entry['direction'])
            
            objs = []
            for o in entry.get('objects_detected', []):
                obj_name = o_map.get(o['object'], o['object'])
                dist = d_map.get(o['distance'], o['distance'])
                objs.append(f"{obj_name}:{dist}")
            
            if objs:
                compressed_parts.append(f"{direction}:({','.join(objs)})")
        
        return " | ".join(compressed_parts)

    def _format_action_schema(self):
        schema = ""
        for action_type in ["continuous", "discrete"]:
            if self.actions.get(action_type):
                for name, details in self.actions[action_type].items():
                    schema += f"- {name}: {details['options']}\n"
        return schema

    def _format_grid_color_schema(self):
        return "".join([f"- {item['color']}: {item['object']}\n" for item in self.grid_color_legend])