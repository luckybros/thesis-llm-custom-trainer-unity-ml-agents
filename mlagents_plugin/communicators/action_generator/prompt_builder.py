import os
import numpy as np

class PromptBuilder:
    def __init__(self, settings):
        
        self.game_overview = settings.game_desc
        self.agent_role = settings.agent_role
        self.task = settings.task
        self.actions = settings.actions
        self.history = []

        
        self.use_vectorial = settings.use_vectorial_obs
        self.use_visual = settings.use_visual_obs
        self.use_raycast = settings.use_raycast_obs
        self.use_grid = settings.use_grid_obs
        if self.use_grid:
            self.grid_color_legend = settings.grid_color_legend

        
        self._initialize_system_message()

    def _initialize_system_message(self):    
        abbreviation_legend = (
            "## DATA ABBREVIATIONS (Mandatory for efficiency)\n"
            "- RF / RB: Raycast Front / Raycast Back\n"
            "- Directions: L (Left), SL (Slight Left), F (Forward), SR (Slight Right), R (Right), FR (Far Right), C (Center)\n"
            "- Objects: OBS (Obstacle), TANK (Enemy), PWUP (PowerUp)\n"
            "- Distances: I (Immediate), C (Close), M (Medium), F (Far)\n"
        )

        self.system_msg = (
            f"### [GAME OVERVIEW]\n{self.game_overview}\n\n"
            f"### [ROLE & TASK]\n{self.agent_role}\nTask: {self.task}\n\n"
            f"{abbreviation_legend}\n"
            f"### [ACTION SET]\n{self._format_action_schema()}\n\n"
        )

        if self.use_grid:
            self.system_msg += f"\n### [GRID LEGEND]\n{self._format_grid_color_schema()}"

    def _compress_raycast(self, raycast_data):
        """Trasforma il JSON pesante in una stringa tecnica ultra-densa."""
        d_map = {'IMMEDIATE': 'I', 'CLOSE': 'C', 'MEDIUM': 'M', 'FAR': 'F'}
        o_map = {'OBSTACLE': 'OBS', 'TANK': 'TANK', 'POWER UP': 'PWUP'}
        dir_map = {'LEFT': 'L', 'SLIGHT_LEFT': 'SL', 'FORWARD': 'F', 
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

    def build_prompt(self, state: dict):
        
        content_text = ""
        if self.use_vectorial:
            vec_obs = state.get("VECTORIAL", {})
            for agent_id, data in vec_obs.items():
                # Rimuoviamo le graffe del dizionario per risparmiare token
                clean_data = str(data).replace("{", "").replace("}", "").replace("'", "")
                content_text += f"[STATUS {agent_id}] {clean_data}\n"

        if self.use_raycast:
            for key, agents_dict in state.items():
                if key.startswith("RAYCAST"):
                    prefix = "RF" if "FRONT" in key else "RB"
                    for agent_id, ray_data in agents_dict.items():
                        compressed = self._compress_raycast(ray_data)
                        content_text += f"[{prefix} {agent_id}] {compressed}\n"

        # 3. Composizione Human Message
        human_text = (
            f"### HISTORY\n{self.history if self.history else 'None'}\n\n"
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

    def _format_action_schema(self):
        schema = ""
        for action_type in ["continuous", "discrete"]:
            if self.actions.get(action_type):
                for name, details in self.actions[action_type].items():
                    schema += f"- {name}: {details['options']}\n"
        return schema
    
    def _format_grid_color_schema(self):
        return "".join([f"- {item['color']}: {item['object']}\n" for item in self.grid_color_legend])