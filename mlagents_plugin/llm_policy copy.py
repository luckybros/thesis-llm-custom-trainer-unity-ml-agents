import numpy as np
import logging

from mlagents.trainers.policy.torch_policy import TorchPolicy
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.action_info import ActionInfo # Importiamo ActionInfo
from mlagents.trainers.behavior_id_utils import get_global_agent_id # Importiamo la funzione mancante
from mlagents.trainers.policy.policy import Policy
from mlagents_envs.logging_util import get_logger
from mlagents_envs.base_env import BehaviorSpec, DecisionSteps, ActionTuple
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.networks import GlobalSteps
from .mock_communication_client import MockCommunicationClient

logger = get_logger(__name__)

class LLMPolicy(Policy):
    def __init__(
            self,
            seed : int,
            behavior_spec: BehaviorSpec,
            network_settings: NetworkSettings
    ):
        
        super().__init__(seed, behavior_spec, network_settings)
        # Implementato in TorchPolicy ma stiamo ereditando da Policy
        self.global_step = (
            GlobalSteps()
        ) 
        self.action_spec = behavior_spec.action_spec
        self.num_actions = behavior_spec.action_spec.discrete_size
        self.num_agents = len(behavior_spec.observation_specs)
        self.communicator_client = MockCommunicationClient(discrete_branches=self.action_spec.discrete_branches, num_agents=self.num_agents)
        logger.info("--- LLM Policy initialized! ---")

    def get_action(
            self, decision_requests: DecisionSteps, worker_id : int
    ) -> ActionInfo:
        
        if len(decision_requests) == 0:
            return ActionInfo.empty()
        
        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ]
        
        obs_list = decision_requests.obs    # un'osservazione per agente
        # masks = self._extract_masks(decision_requests)

        # Here the llm will get the observation and choose an action, for now let's choose one randomly
        #action = self.action_spec.random_action(len(decision_requests))

        # Voglio ottenere una lista di distribuzioni di probabilità, una per azione
        (action_dists, user) = self.communicator_client.recieve_action_from_llm(obs_list)

        final_action = []
        for i in range(self.num_agents):
            agent_action = []
            distribution_for_agent = action_dists[i]
            for dist in distribution_for_agent:
                value = np.random.choice(len(dist), p=dist)
                agent_action.append(value)
            final_action.append(agent_action)

        action = ActionTuple()
        action.add_discrete(np.array(final_action))
    
        outputs = {"action": action}
        if user == 1:
            print(action.discrete)

        return ActionInfo(
            action=action,
            env_action=action,
            outputs=outputs,
            agent_ids=list(decision_requests.agent_id),
        )
    
    def get_current_step(self):
        """
        Astratto da implementare, implementato in TorchPolicy
        """
        return self.global_step.current_step

        
