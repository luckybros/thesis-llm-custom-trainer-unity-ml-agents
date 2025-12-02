from typing import Dict, List
import numpy as np
from mlagents_plugin.communicators.action_generator.llm_action_generator import LLMActionGenerator
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
import re

load_dotenv()

class LangchainActionGenerator(LLMActionGenerator):
    
    def __init__(self, discrete_branches: tuple[int], num_continuous_action: int, num_agents: int):
        super().__init__(discrete_branches, num_continuous_action, num_agents)
        # All these parameters must go in a yaml 
        self.game_desc = """Basic is a simple game where the objective is to control a
        small cube-shaped character to collect spheres and achieve the highest 
        possible score. At each iteration, it is possible to move in various 
        directions: there is a large sphere on the right and a small one on the left. """
        self.agent_role = f"""You are a game AI assistant deployed in the testing 
        environment of Basic. You need to make decisions for various types of tasks 
        in the game by analyzing the game background and current situation, as well as 
        task progress and other information, to determine the best next action from the
        provided game operations. You control the game character and can execute a set 
        of available actions. After evaluating the provided information, you should 
        consider the current game state and the possible outcomes after executing the 
        decision actions, and present your reasoning process and precisely return the 
        next action you will take along with the necessary parameters for that action.
        You shoud choose actions for {self.num_agents} agents
        In every observation, the 1 represent the agent position, and the 0 the other
        positions."""
        self.history_length = 3
        self.model_name = "gemini-2.5-flash" 

        self.task = "Obtain the highest score"
        self.actions = ["Go left", "Do nothing", "Go right"]
        self.history : List[str] = []   # it could make sense have an history for agent

        # Configuratin messages
        system_msg = f"""
            [Game Overview] \n {self.game_desc} \n\n",
            [Role and function] \n {self.agent_role}"
        """
        human_msg = """
            [Current Task] \n {task} \n\n", # This should be extended to task information but for now it's a string
            [Action Set] \n Available actions: {actions} \n\n", # Every action set should be a dict 
            [History] \n {history} \n\n",
            [Current Observations] \n {observations} \n\n",
            Given these information, return the next action you would take."
        """
        
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", system_msg),
            ("human", human_msg)
        ])

        self.model = ChatGoogleGenerativeAI(model=self.model_name)

        self.chain = self.prompt_template | self.model | StrOutputParser()

    def update_history(self, observation: str, action: str):
        self.history.append(f"Observation: {observation}. Action: {action}")
        if len(self.history) > self.history_length:
            self.history.pop(0)

    def encode_state(self, state):
        return state
    
    def get_llm_policy(self, state):
     # we should obtain informations about action from the game, since we have information about the num of continuous actions and discrete. For now let's use a mock
        payload = {}
        dist_distr = []
        for agent, obs in state.items():
            # costruire prompt
            llm_choice = self._generate_action(obs)
            # estrarre azione
            action = self._extract_action(llm_choice)
            # generare distribuzione di probabilità con 1 per quell'azione (per il discreto)
            dist = np.zeros(len(self.actions))
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
            

    def _generate_action(self, obs):
        return self.chain.invoke({
            "task": self.task, 
            "actions": self.actions,
            "history": self.history,
            "observations": obs
        })
    
    def _extract_action(self, output_text):
        lines = output_text.strip().split('\n')
        for line in reversed(lines):  # scorri dal basso (spesso action alla fine)
            for action in self.actions:
                pattern = re.compile(rf"\b{re.escape(action)}\b", re.IGNORECASE)
                if pattern.search(line):
                    return self.actions.index(action)
    

        

