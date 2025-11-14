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
from .mock_communication_client import MockCommunicationClient

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
        logger.info("--- LLM Policy initialized! ---")

    def get_action(
            self, decision_requests: DecisionSteps, worker_id : int
    ) -> ActionInfo:
        
        action_info = super.get_action(decision_requests, worker_id)
        
        llm_run_out = self.llm_evaluate(decision_requests)  # Dovrà contenere l'azione scelta dall'LLM e le distribuzioni di probabilità

        action_info.outputs["llm_action"] = llm_run_out["action"]
        action_info.outputs["llm_log_probs"] = llm_run_out["log_probs"]

        return action_info


    def llm_evaluate(
            self, decision_requests: DecisionSteps
    ) -> Dict[str, Any]:
        
        obs = decision_requests.obs
        masks = self._extract_masks(decision_requests)

        # Le azioni devono essere di tipo AgentAction, mentre le distribuzioni di tipo DistInstances, e le log_probs di tipo ActionLogProbs (bisogna scrivere un util sicuramente)
        llm_action, llm_run_out = self.communicator_client.recieve_action_from_llm(obs) # AgentAction, Dict[str, Any]

        llm_run_out["action"] = llm_action.to_action_tuple()

        if "log_probs" in llm_run_out:
            llm_run_out["log_probs"] = llm_run_out["log_probs"].to_log_probs_tuple()

        return llm_run_out
        
        
