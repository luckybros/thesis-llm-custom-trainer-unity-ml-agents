from typing import Dict, List
import numpy as np
from mlagents_plugin.communicators.action_generator.llm_action_generator import LLMActionGenerator
from mlagents_plugin.communicators.action_generator.llm_action_generator_settings import LLMActionGeneratorSettings
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
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
        self.settings = LLMActionGeneratorSettings(settings_path=settings_path)
        self.history : List[str] = []   # it could make sense have an history for agent

        # Configuratin messages
        system_msg = f"""
            [Game Overview] \n {self.settings.game_desc} \n\n",
            [Role and function] \n {self.settings.agent_role}"
        """
        human_msg = """
            [Current Task] \n {task} \n\n", 
            [Action Set] \n Available actions: {actions} \n\n", 
            [History] \n {history} \n\n",
            [Current Observations] \n {observations} \n\n",
            Given these information, return the next action you would take."
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", human_msg)
        ])

        self.model = ChatGoogleGenerativeAI(model=self.settings.model_name)

        self.chain = self.prompt_template | self.model | StrOutputParser()

    def update_history(self, observation: str, action: str):
        self.history.append(f"Observation: {observation}. Action: {action}")
        if len(self.history) > self.settings.history_length:
            self.history.pop(0)

    def encode_state(self, state):
        return state
    
    def get_llm_policy(self, state):
     # we should obtain informations about action from the game, since we have information about the num of continuous actions and discrete. For now let's use a mock
        payload = {}
        dist_distr = []

        llm_choice = self._generate_actions(state)
        print(llm_choice)

        # estrarre azione
        actions = self._extract_actions(llm_choice)
        # generare distribuzione di probabilità con 1 per quell'azione (per il discreto)
        payload = {}
        if len(self.discrete_branches) > 0:
            payload["discrete"] = self._generate_discrete_distributions(actions)
        return payload
        """
            print(llm_choice)
            # estrarre azione
            action = self._extract_action(llm_choice)
            
            dist = np.zeros(len(self.settings.actions))
            dist[action] = 1.0
            dist_distr.append(dist)
            # aggiornare history
        distributions = {}
        distributions['agent_0-0'] = dist_distr
        distributions = {
            k: [arr.tolist() for arr in v]
            for k, v in distributions.items()
        }
        payload["discrete"] = distributions
        return payload
        """
            

    def _generate_actions(self, obs):
        # the call at the llm should give a choice option for every branch
        print(f"obs: {obs}")
        return self.chain.invoke({
            "task": self.settings.task, 
            "actions": self.settings.actions,
            "history": self.history,
            "observations": obs
        })
    
    def _extract_actions(self, output_text: str) -> List[int]:
        """
        agent_regex = r'Agent\s*(\d+):\s*\[(.+?)\]'
        matches = re.findall(agent_regex, output_text, re.IGNORECASE)
        agent_actions = [[1 for size in self.settings.actions] for _ in range(self.num_agents)]  # default: Do nothing, Don't hit...
        for idx_str, acts_str in matches:
            idx = int(idx_str)
            act_items = [a.strip() for a in acts_str.split(',')]
            if idx < self.num_agents and len(act_items) == len(self.settings.actions):
                agent_actions[idx] = [
                    next(
                        (i for i, legal in enumerate(branch) if a.lower() == legal.lower()),
                        1  # default index se non trovato (es. Do nothing, o Don't hit)
                    )
                    for a, branch in zip(act_items, self.settings.actions)
                ]
            # else fallback/skip, puoi aggiungere log/warning
        return agent_actions
        """
        agent_regex = r'Agent\s*(\d+):\s*\[(.+?)\]'
        matches = re.findall(agent_regex, output_text, re.IGNORECASE)
        print(f"Matches: {matches}")
        agent_actions = [[0 for size in self.settings.actions] for _ in range(self.num_agents)]
        print(f"agent_action: {agent_actions}")
        for idx_str, acts_str in matches:
            print(f"idx: {idx_str}, acts_str: {acts_str}")
            idx = int(idx_str)
            act_items = [a.strip() for a in acts_str.split(',')]    # list
            for i, act in enumerate(act_items):
                agent_actions[idx][i] = self.settings.actions[i].index(act)
        print(agent_actions)
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


        

