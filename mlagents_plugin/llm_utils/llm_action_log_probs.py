from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from typing import List, Optional, NamedTuple
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents_envs.base_env import _ActionTupleBase
from .llm_utils.llm_buffer import LLMBufferKey

class LLMActionLogProbs(ActionLogProbs):

    def from_buffer_llm(buff: AgentBuffer) -> "ActionLogProbs":
        """
        A static method that accesses continuous and discrete log probs fields from the 
        LLM in an AgentBuffer and constructs the corresponding ActionLogProbs from the retrieved np arrays.
        """
        continuous: torch.Tensor = None
        discrete: List[torch.Tensor] = None  # type: ignore

        if LLMBufferKey.LLM_CONTINUOUS_LOG_PROBS in buff:
            continuous = ModelUtils.list_to_tensor(buff[BufferKey.CONTINUOUS_LOG_PROBS])
        if LLMBufferKey.LLM_DISCRETE_LOG_PROBS in buff:
            discrete_tensor = ModelUtils.list_to_tensor(
                buff[BufferKey.DISCRETE_LOG_PROBS]
            )
            # This will keep discrete_list = None which enables flatten()
            if discrete_tensor.shape[1] > 0:
                discrete = [
                    discrete_tensor[..., i] for i in range(discrete_tensor.shape[-1])
                ]
        return LLMActionLogProbs(continuous, discrete, None)
    
    def flatten_all_discrete(self) -> torch.Tensor:
        """
        A utility method that returns all log probs
        """


