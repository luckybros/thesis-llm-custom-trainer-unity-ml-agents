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
            [Current Task] \n {task} \n\n, 
            [Action Set] \n Available actions: {actions} \n\n, 
            [History] \n {history} \n\n,
            [Vect observations] \n {observations} \n\n,
            Given these information, return the next action you would take."
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
        print(llm_choice)

        # estrazione dell'azione all'LLM
        actions = self._extract_actions(llm_choice)
        # generare distribuzione di probabilità con 1 per quell'azione (per il discreto)
        payload = {}
        if len(self.discrete_branches) > 0:
            payload["discrete"] = self._generate_discrete_distributions(actions)
        return payload

            
    def _generate_actions(self, obs):
        # the call at the llm should give a choice option for every branch
        return self.chain.invoke(obs)
    
    def _extract_actions(self, output_text: str) -> List[int]:
        """
        Takes the str of the output of the LLM, and gives a list of list, the same
        size of the number of action (discrete actions) and at every index it has
        the number of the action chosen for that agent
        """
        agent_regex = r'Agent\s*(\d+):\s*\[(.+?)\]'
        matches = re.findall(agent_regex, output_text, re.IGNORECASE)
        print(f"Matches: {matches}")
        agent_actions = [[0 for size in self.settings.actions] for _ in range(self.num_agents)]
        for idx_str, acts_str in matches:
            print(f"idx: {idx_str}, acts_str: {acts_str}")
            idx = int(idx_str)
            act_items = [a.strip() for a in acts_str.split(',')]    # list
            print(f"act_items: {act_items}")
            if len(act_items) != len(self.settings.actions):
                continue 
            for i, act in enumerate(act_items):
                if act not in self.settings.actions[i]:
                    continue
                agent_actions[idx][i] = self.settings.actions[i].index(act)
        # print(agent_actions)
        return agent_actions
    
    def _generate_discrete_distributions(self, actions):
        distributions = {}
        # [[2, 0], [2, 1]] agent, action
        for idx, action in enumerate(actions):
            # 0, [2, 0]
            dists_for_agent = [np.zeros(size) for size in self.discrete_branches]
            for i, a in enumerate(action):
                dists_for_agent[i][a] = 1.0
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
            actions=self.settings.actions,
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
        
        return [
            SystemMessage(content=self.system_msg),
            HumanMessage(content=content)
        ]


        

