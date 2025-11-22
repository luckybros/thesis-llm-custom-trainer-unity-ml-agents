from typing import Any, Dict, List
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
from mlagents_plugin.communicators.mock_communication_client import MockCommunicationClient
from mlagents_plugin.trainers.llm_buffer import LLMBuffer, LLMBufferKey

logger = get_logger(__name__)

class TorchLLMPolicy(TorchPolicy):
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings: NetworkSettings,
        actor_cls: type,
        actor_kwargs: Dict[str, Any],
    ):
        
        super().__init__(seed, behavior_spec, network_settings, actor_cls, actor_kwargs)
        
        self.action_spec = behavior_spec.action_spec
        self.num_actions = behavior_spec.action_spec.discrete_size
        self.num_agents = len(behavior_spec.observation_specs)
        self.communicator_client = MockCommunicationClient(discrete_branches=self.action_spec.discrete_branches, num_agents=self.num_agents)
        self.llm_buffer = LLMBuffer()
        

    def get_action(
            self, decision_requests: DecisionSteps, worker_id : int
    ) -> ActionInfo:
        
        llm_run_out = self.llm_evaluate(decision_requests)  

        if "discrete" in llm_run_out:
            discrete_log_probs = llm_run_out["discrete"]
            for log_prob in discrete_log_probs:
                self.llm_buffer.add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, log_prob)
        
        if "continuous" in llm_run_out:
            continuous_log_probs = llm_run_out["continuous"]
            for log_prob in continuous_log_probs:
                self.llm_buffer.add_entry(LLMBufferKey.LLM_LOG_CONTINUOUS_LOG_PROBS, log_prob)
        
        return super().get_action(decision_requests, worker_id)


    def llm_evaluate(
            self, decision_requests: DecisionSteps
    ) -> Dict[str, List[np.ndarray]]:
        """
        Interrogate the LLM and obtains action log probs distribution in this format
        {
            "discrete": [array_ag1, array_ag2, ...],
            "continuous": [array_ag1, array_ag2, ...]
        }
        """
        obs = decision_requests.obs
        masks = self._extract_masks(decision_requests)

        # Le azioni devono essere di tipo AgentAction, mentre le distribuzioni di tipo DistInstances, e le log_probs di tipo ActionLogProbs (bisogna scrivere un util sicuramente)
        llm_run_out = self.communicator_client.recieve_action_from_llm(obs) # Dict[str, Any]

        return llm_run_out
        
    def pop_llm_buffer_data(
            self, num_items: int
    ) -> Dict[LLMBufferKey, List[np.ndarray]]:
        return self.llm_buffer.pop_n_entries(num_items=num_items)
