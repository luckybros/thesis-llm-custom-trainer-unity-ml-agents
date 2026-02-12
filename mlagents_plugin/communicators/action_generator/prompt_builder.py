
class PromptBuilder:
    def __init__(self, settings):
        
        self.task = settings.task
        self.actions = settings.actions
        self.history = []   # devo vedere come salvare l'azione presa dal modello nella storia, per ora lista vuota
    
        self.use_vectorial = settings.use_vectorial_obs
        self.use_visual = settings.use_visual_obs
        self.use_raycast = settings.use_raycast_obs
        self.use_grid = settings.use_grid_obs
        if self.use_grid:
            self.grid_color_legend = settings.grid_color_legend

        self.system_msg = f"""
            [Game Overview] \n {settings.game_desc}, \n\n
            [Role and function] \n {settings.agent_role} \n\n
            [Action Set] \n {self._format_action_schema()}, \n\n
            [Task Information] \n {self.task} \n\n
        """

        # lo faccio una sola volta poiché la leggenda di colori è sempre la stessa
        if self.use_grid:
            self.system_msg += f"[Color legend for grid images] \n {self._format_grid_color_schema()} \n\n"

        # action set ora è fisso ma potrebbe cambaire con l'ottimizzatore
        # stessa cosa per il task
        self.human_mgs = """
            [History] \n {history}, \n\n
            [Useful information to help decision making] \n {observations}, \n\n
        """

    def build_prompt(self, state: dict):
        """
        Format the input data for the LLM.

        Args:
            data (dict): The input data containing the keys "states", "images", etc.

        Returns:
            list: A list of SystemMessage and HumanMessage formatted for the LLM.
        """

        content = []
        content_text = ""

        if self.use_vectorial: 
            observation_text = state.get("VECTORIAL", "None") # dovremmo chiamarlo vectorial poiché questi in realtà sono gli astratti del vettoriale
            content_text += str(observation_text)
            content_text += "\n"
        
        if self.use_raycast:
            raycast_text = state.get("RAYCAST", "None")
            content_text += str(raycast_text)
            content_text += "\n"

        final_text = self.human_mgs.format(
            history = self.history,
            observations = content_text
        )

        # prompt per ora rispettando logica langchain, poi vediamo per il locale
        content.append({
            "type": "text",
            "text": final_text
        })

        # images
        if self.use_visual:
            images = state.get("VISUAL", {})
            for agent_id, img_b64 in images.items():
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })

        # grid
        if self.use_grid:
            images = state.get("GRID", {})
            for agent_id, img_b64 in images.items():
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })

        result = {"sys_msg": self.system_msg, "hum_msg": content}
        #print(f"prompt: {result}")
        return result


    def _format_action_schema(self):
        """
        Crea una stringa formattata per l'LLM che mostra SOLO nomi e opzioni.
        NASCONDE i valori numerici.
        Nel prompt non facciamo distinzioni tra continue e discrete
        """
        schema = "---Actions---"
        
        if self.actions.get("continuous"):
            #schema += "--- Continuous Actions ---\n"
            for name, details in self.actions["continuous"].items():
                desc = f" ({details['description']})" if details.get('description') else ""
                schema += f"- **{name}**{desc}: {details['options']}\n"

        if self.actions.get("discrete"):
            #schema += "\n--- Discrete Actions ---\n"
            for name, details in self.actions["discrete"].items():
                desc = f" ({details['description']})" if details.get('description') else ""
                schema += f"- **{name}**{desc}: {details['options']}\n"
                
        return schema
    
    def _format_grid_color_schema(self):
        result = []
        for item in self.grid_color_legend:
            line = f"- **{item['color']}**: {item['object']}"
            if 'desc' in item:
                line += f" ({item['desc']})"
            result.append(line)
        return "".join(result)