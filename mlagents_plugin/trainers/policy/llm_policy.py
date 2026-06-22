from typing import Any, Dict, List
import numpy as np
import logging, random
from mlagents.torch_utils import torch, default_device

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
from mlagents_envs.timers import timed
#from mlagents_plugin.communicators.client.mock_communication_client import MockCommunicationClient
from mlagents_plugin.communicators.client.random_communication_client import RandomCommunicationClient
from mlagents_plugin.trainers.llm_buffer import LLMBuffer, LLMBufferKey
from mlagents_plugin.utils.llm_utils import LLMUtils
from mlagents.trainers.torch_entities.utils import ModelUtils

logger = get_logger(__name__)

class TorchLLMPolicy(TorchPolicy):

    GLOBAL_LLM_BUFFERS: Dict[str, LLMBuffer] = {}
    def __init__(
        self,
        seed: int,
        behavior_spec: BehaviorSpec,
        network_settings: NetworkSettings,
        actor_cls: type,
        actor_kwargs: Dict[str, Any],
        communicator_cls: type,
        llm_refresh_interval: int,
        observation_types: list,
    ):
        
        super().__init__(seed, behavior_spec, network_settings, actor_cls, actor_kwargs)
        
        self.action_spec = behavior_spec.action_spec
        self.num_actions = behavior_spec.action_spec.discrete_size
        self.num_continuous_action = behavior_spec.action_spec.continuous_size
        self.num_agents : int = None
        self._is_discrete : bool = behavior_spec.action_spec.discrete_size > 0
        self._is_continuous : bool = behavior_spec.action_spec.continuous_size > 0
        # qua si puo fare cls nel costruttore e specificare il client da iperparametro
        # logger.info(f"observation_specs : {behavior_spec.observation_specs}")
        self._communicator_cls = communicator_cls
        self.communicator_client = None
        #self.agent_llm_buffers: Dict[str, LLMBuffer] = {}
        self.llm_refresh_interval = llm_refresh_interval
        self._llm_step_counter = 0
        self.observation_types = observation_types
        self._is_ghost_frozen: bool = False

    def set_ghost_frozen(self, frozen: bool) -> None:
        """
        Imposta la policy in modalità ghost (freezata) o learning (attiva).
        Quando frozen=True, get_action salta la chiamata all'LLM e usa solo la rete neurale.
        """
        #logger.info(f"🧊 [GHOST DEBUG] Policy ID {id(self)} - Cambio stato frozen: da {self._is_ghost_frozen} a {frozen}")
        self._is_ghost_frozen = frozen

    def get_action(
            self, decision_requests: DecisionSteps, worker_id : int
    ) -> ActionInfo:
        # LLM part
        obs = decision_requests.obs
                    
        global_agent_ids = [
            get_global_agent_id(worker_id, int(agent_id))
            for agent_id in decision_requests.agent_id
        ] if self.num_agents != 1 else ['agent_0-0']
        # Don't know why in basic the global ids change, so force them to the same
        #logger.info(f"llm_run_out: {llm_run_out}")
        refresh_llm = self._llm_step_counter % self.llm_refresh_interval == 0
        #print(refresh_llm)
        #logger.info(f"global_agent_ids: {global_agent_ids}")

        # The problem is that process trajectory can't have missing piece of information, so we have
        # to add empty lists when we don't call the llm
        if refresh_llm and not self._is_ghost_frozen:
            selected_indices = list(range(len(global_agent_ids)))

            llm_run_out_batch = self.llm_evaluate_batch(decision_requests, global_agent_ids, selected_indices)

            for agent_id in global_agent_ids:
                llm_run_out = llm_run_out_batch[agent_id]

                if "discrete" in llm_run_out:
                    discrete_log_probs = llm_run_out["discrete"]
                    if agent_id not in TorchLLMPolicy.GLOBAL_LLM_BUFFERS:
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id] = LLMBuffer()
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, discrete_log_probs['agent_0-0'])
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 1)
            """
            for agent_id in global_agent_ids:
                selected_index = global_agent_ids.index(agent_id)
                llm_run_out = self.llm_evaluate(decision_requests, selected_index)
                if "discrete" in llm_run_out:
                    discrete_log_probs = llm_run_out["discrete"]
                    if agent_id not in TorchLLMPolicy.GLOBAL_LLM_BUFFERS:
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id] = LLMBuffer()
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, discrete_log_probs['agent_0-0'])
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 1)
            """

            """
            selected_agent_id = random.choice(global_agent_ids)
            selected_index = global_agent_ids.index(selected_agent_id)
            llm_run_out = self.llm_evaluate(decision_requests, selected_index)  
            if "discrete" in llm_run_out:
                discrete_log_probs = llm_run_out["discrete"]
                logger.info(f"Discrete log probs: {discrete_log_probs}")
                for agent_id in global_agent_ids:
                    if agent_id not in TorchLLMPolicy.GLOBAL_LLM_BUFFERS:
                        #logger.info(f"{agent_id} not in llm_buffer, creating a new one")
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id] = LLMBuffer()
                        #self.agent_llm_buffers[agent_id] = LLMBuffer()
                    if agent_id == selected_agent_id:
                        #logger.info(f"Adding...")
                        #logger.info(f"agent_id: {agent_id}")
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, discrete_log_probs['agent_0-0'])
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 1)
                        #self.agent_llm_buffers[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, discrete_log_probs['agent_0-0'])
                        #self.agent_llm_buffers[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 1)
                    else:
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, [])
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 0)   
            """

            """
                for agent_id, dist in discrete_log_probs.items():
                    if agent_id not in self.agent_llm_buffers:
                        self.agent_llm_buffers[agent_id] = LLMBuffer()
                    # Squeeze nel caso single action, si puo fare sicuramente meglio 
                    # dist = LLMUtils.squeeze_list_dim(batch_list=dist)
                    logger.info(f": {agent_id}")
                    logger.info(f"Selected agent for LLM: {selected_agent_id}")
                    if agent_id == selected_agent_id:    
                        self.agent_llm_buffers[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, dist)
                        self.agent_llm_buffers[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 1)
                    else:
                        self.agent_llm_buffers[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, [])
                        self.agent_llm_buffers[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 0)
            """

            """
            if "continuous" in llm_run_out:
                continuous_log_probs = llm_run_out["continuous"]
                for agent_id, dist in continuous_log_probs.items():
                    if agent_id not in TorchLLMPolicy.GLOBAL_LLM_BUFFERS:
                        TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id] = LLMBuffer()
                    #logger.info(f"cccc {agent_id}")
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_LOG_CONTINUOUS_LOG_PROBS, dist)
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_MASK_CONTINUOUS, 1)
            """

            self._llm_step_counter = 0

        else:
            for agent_id in global_agent_ids:
                if agent_id not in TorchLLMPolicy.GLOBAL_LLM_BUFFERS:
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id] = LLMBuffer()
                #logger.info(f"eeee {agent_id}")
                # Discrete
                if self._is_discrete > 0:
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, [])
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_MASK_DISCRETE, 0)
                # Continuous
                if self._is_continuous > 0:
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_LOG_CONTINUOUS_LOG_PROBS, [])
                    TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].add_entry(LLMBufferKey.LLM_MASK_CONTINUOUS, 0)
                
        self._llm_step_counter += 1

        return super().get_action(decision_requests, worker_id)
    
    def llm_evaluate(
            self, decision_requests: DecisionSteps, selected_index
    ) -> Dict[str, List[np.ndarray]]:
        """
        Interrogate the LLM and obtains action log probs distribution in this format
        {
            "discrete": [array_ag1, array_ag2, ...],
            "continuous": [array_ag1, array_ag2, ...]
        }
        """
        obs = decision_requests.obs
        num_agents = len(obs[0])

        if self.num_agents is None:
            self.num_agents = num_agents

        if self.communicator_client is None:
            self.communicator_client = self._communicator_cls(
                discrete_branches=self.action_spec.discrete_branches, 
                num_continuous_action=self.num_continuous_action, 
                num_agents=1, 
                observation_types=self.observation_types
            )
        masks = self._extract_masks(decision_requests)

        llm_run_out = self.communicator_client.receive_distribution_from_llm(obs, selected_index) # Dict[str, Any]
        return llm_run_out
        
    def llm_evaluate_batch(
            self, decision_requests: DecisionSteps, agent_ids: list, selected_indices: list
    ) -> dict:
        """
        Interrogate LLM, but only one time instead of one call for agent

        Ritorna: dict {agent_id: llm_run_out}, dove ogni llm_run_out ha lo
             stesso formato che llm_evaluate ritornava per un singolo
             agente:
             {
                 "discrete": {...},
                 "continuous": {...}
             }
        """
        obs = decision_requests.obs
        num_agents = len(obs[0])

        if self.num_agents is None:
            self.num_agents = num_agents

        if self.communicator_client is None:
            self.communicator_client = self._communicator_cls(
                discrete_branches=self.action_spec.discrete_branches,
                num_continuous_action=self.num_continuous_action,
                num_agents=1,
                observation_types=self.observation_types
            )

        masks = self._extract_masks(decision_requests)

        llm_run_out_batch = self.communicator_client.receive_distribution_from_llm_batch(
            obs, selected_indices
        )

        return llm_run_out_batch
            
    def pop_llm_buffer_data(
            self, agent_id: int, num_items: int
    ) -> Dict[LLMBufferKey, List[np.ndarray]]:

        # Hard-coded per ora su Basic, poihé non so perché agent_id cambia sempre se ho un agente
        if self.num_agents == 1:
            #logger.info(f"superdebug: {self.agent_llm_buffers['agent_0-0']}")
            #return self.agent_llm_buffers["agent_0-0"].pop_n_entries(num_items=num_items)
            return TorchLLMPolicy.GLOBAL_LLM_BUFFERS["agent_0-0"].pop_n_entries(num_items=num_items)
        
        if agent_id not in TorchLLMPolicy.GLOBAL_LLM_BUFFERS:
            logger.error(f"{agent_id} NON esiste in questo dizionario!")
            return {}

        return TorchLLMPolicy.GLOBAL_LLM_BUFFERS[agent_id].pop_n_entries(num_items=num_items)