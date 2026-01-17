from typing import Dict, List
import numpy as np
from mlagents_plugin.communicators.action_generator.llm_action_generator import LLMActionGenerator
from mlagents_plugin.communicators.action_generator.langchain_action_generator_settings import LangchainActionGeneratorSettings
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
#from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import re

load_dotenv()

class LangchainActionGenerator(LLMActionGenerator):
    
    def __init__(
            self, 
            discrete_branches: tuple[int], 
            num_continuous_action: int, 
            num_agents: int,
            settings_path: str):
        super().__init__(discrete_branches, num_continuous_action, num_agents)
        self.settings = LangchainActionGeneratorSettings(settings_path=settings_path)
        self.history : List[str] = []   # it could make sense have an history for agent

        # Configuratin messages
        self.system_msg = f"""
            [Game Overview] \n {self.settings.game_desc} \n\n,
            [Role and function] \n {self.settings.agent_role} \n\n
        """
        # aggiungere nel template immagini

        self.human_msg = """
            [History] \n {history} \n\n,
            [Action Set] \n Available actions: {actions} \n\n, 
            [Useful information to help decision making] \n {observations} \n\n"
            [Task Information] \n {task} \n\n, 
        """

        

        """
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", human_msg)
        ])
        """

        self.model = self.settings.model_constructor_generator()

        # Chain dinamica che compone il messaggio a seconda del fatto che sia solo visual, visual + vect o solo vect
        self.chain = RunnableLambda(self._format_input) | self.model | StrOutputParser()

    def update_history(self, observation: str, action: str):
        self.history.append(f"Observation: {observation}. Action: {action}")
        if len(self.history) > self.settings.history_length:
            self.history.pop(0)

    def encode_state(self, state):
        return state
    
    def get_llm_policy(self, state):
        # we should obtain informations about action from the game, since we have information about the num of continuous actions and discrete. For now let's use a mock
        # ottenere visuali e vettoriali
        # devo modificare questa cosa
        # magari aggiungere un'iperparametro che batcha 

        # prompt all'LLM
        llm_choice = self._generate_actions(state)
        print(f"llm_choice: {llm_choice}")

        # estrazione dell'azione all'LLM
        result_actions = self._extract_actions(llm_choice)
        print(f"result action: {result_actions}")
        # generare distribuzione di probabilità con 1 per quell'azione (per il discreto)
        payload = {}
        if len(self.discrete_branches) > 0:
            payload["discrete"] = self._generate_discrete_distributions(result_actions)
        if self.num_continuous_action > 0:
            payload["continuous"] = self._generate_continuous_distribuitons(result_actions)

        print(f"payload: {payload}")
        return payload

            
    def _generate_actions(self, obs):
        # the call at the llm should give a choice option for every branch
        return self.chain.invoke(obs)
    
    def _extract_actions(self, output_text: str) -> Dict:
        """
        Agent 0:
            continuous:
                Z Rotation (Steering)
                    Rotate left
                X Rotation (Tilt)
                    Rotate backward
        Agent 1:
            continuous:
                Z Rotation (Steering)
                    Rotate left
                X Rotation (Tilt)
                    Rotate backward

        Return Structure:
        {
            0: {  # Agent ID
                "continuous": {
                    "Z Rotation (Steering)": "Rotate left",
                    "X Rotation (Tilt)": "Rotate backward"
                },
                "discrete": {
                    "Move": "Right"
                }
            },
            ...
        }
        """
        result = {}
        matches = re.findall(r'Agent\s+(\d+):(.*?)(?=Agent\s+\d+:|$)', output_text, re.DOTALL)
        if len(matches) > self.num_agents:
            print("aggiustando matches")
            matches = matches[self.num_agents:]
        print(f"Matches: {matches}")
        # agent_actions = [[0 for size in self.settings.actions] for _ in range(self.num_agents)]
        for idx_str, acts_str in matches:
            print(f"idx: {idx_str}, acts_str: {acts_str}")
            idx = int(idx_str)
            result[idx] = {"continuous" : {}, "discrete": {}}

            # Continuous extraction
            for act_name in self.settings.actions["continuous"].keys():
                pattern = re.escape(act_name) + r'\s*\n\s*(.+)'
                match = re.search(pattern, acts_str)

                if match:
                    choice = match.group(1).strip()
                    result[idx]["continuous"][act_name] = choice

            for act_name in self.settings.actions["discrete"].keys():
                pattern = re.escape(act_name) + r'\s*\n\s*(.+)'
                match = re.search(pattern, acts_str)

                if match:
                    choice = match.group(1).strip()
                    result[idx]["discrete"][act_name] = choice

        return result
    
    def _generate_discrete_distributions(self, result: Dict) -> Dict:
        """
        {
            0: {  # Agent ID
                "continuous": {
                    "Z Rotation (Steering)": "Rotate left",
                    "X Rotation (Tilt)": "Rotate backward"
                },
                "discrete": {
                    "Move": "Right"
                }
            },
        }
        """
        distributions = {}

        for idx, agent_values in enumerate(result.values()):

            dists_for_agent = [np.zeros(size) for size in self.discrete_branches]
            # one for agent
            discrete_value = agent_values['discrete']
            """
            {
                'Z Rotation (Steering)': 'Rotate left', 
                'X Rotation (Tilt)': 'Rotate forward'
            }
            """
            for i, (k, v) in enumerate(discrete_value.items()):
                # k->Z Rotation (Steering), v->Rotate left
                indx = self.settings.get_index_of_action(k, v, False)
                dists_for_agent[i][indx] = 1.0
            distributions[f"agent_0-{idx}"] = dists_for_agent

        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }

        return distributions
    
    def _generate_continuous_distribuitons(self, result):
        """
        {
            0: {  # Agent ID
                "continuous": {
                    "Z Rotation (Steering)": "Rotate left",
                    "X Rotation (Tilt)": "Rotate backward"
                },
                "discrete": {
                    "Move": "Right"
                }
            },
        }
        """
        distributions = {}

        for idx, agent_values in enumerate(result.values()):

            dists_for_agent = [np.zeros(2) for _ in range(self.num_continuous_action)] 
            # one for agent
            continuous_value = agent_values['continuous']
            """
            {
                'Z Rotation (Steering)': 'Rotate left', 
                'X Rotation (Tilt)': 'Rotate forward'
            }
            """
            for i, (k, v) in enumerate(continuous_value.items()):
                # k->Z Rotation (Steering), v->Rotate left
                indx = self.settings.get_index_of_action(k, v, True)
                value = self.settings.get_continuous_value_by_index(k, indx)
                dists_for_agent[i][0] = value
            distributions[f"agent_0-{idx}"] = dists_for_agent

        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }

        return distributions

    def _format_input(self, data: dict):
        """
        Format the input data for the LLM.

        Args:
            data (dict): The input data containing the keys "states", "images", etc.

        Returns:
            list: A list of SystemMessage and HumanMessage formatted for the LLM.
        """
        # deve essere come il prompt template, nel senso che riempie system_msg e human_msg
        # con le osservazioni, solo che a differenza del prompt template dobbiamo mettere 
        # anche le immagini
        content = []

        # vectorial
        observation_text = data.get("states", "None, only visuals")

        final_text = self.human_msg.format(
            task=self.settings.task,
            actions=self._format_action_schema(),
            history=self.history,
            observations=observation_text
        )
        content.append({
            "type": "text",
            "text": final_text
        })

        # images
        if self.settings.use_visual_obs:
            images = data.get("images", {})
            for agent_id, img_b64 in images.items():
                content.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/png;base64,{img_b64}"}
                })
        
        """
        sys_msg = SystemMessage(content=self.system_msg)
        hum_msg = HumanMessage(content=content)
        msg = [sys_msg, hum_msg]
        print(f"Messaggio: {msg}")
        return msg
        """
        return [
            SystemMessage(content=self.system_msg),
            HumanMessage(content=content)
        ]

    def _format_action_schema(self):
        """
        Crea una stringa formattata per l'LLM che mostra SOLO nomi e opzioni.
        NASCONDE i valori numerici.
        """
        schema = ""
        
        if self.settings.actions.get("continuous"):
            schema += "--- Continuous Actions ---\n"
            for name, details in self.settings.actions["continuous"].items():
                desc = f" ({details['description']})" if details.get('description') else ""
                schema += f"- **{name}**{desc}: {details['options']}\n"

        if self.settings.actions.get("discrete"):
            schema += "\n--- Discrete Actions ---\n"
            for name, details in self.settings.actions["discrete"].items():
                desc = f" ({details['description']})" if details.get('description') else ""
                schema += f"- **{name}**{desc}: {details['options']}\n"
                
        return schema


        

