from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from typing import List, Optional, NamedTuple
from mlagents.torch_utils import torch
import numpy as np

from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents.trainers.buffer import AgentBuffer, BufferKey
from mlagents_envs.base_env import _ActionTupleBase
from mlagents_plugin.utils.llm_buffer import LLMBufferKey

class LLMActionLogProbs(ActionLogProbs):

    @staticmethod
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
    
    def _all_distributions_to_tensor_list(self) -> List[torch.Tensor]:
        """
        Returns the distributions (both continue and discrete) in the ActionLogProbs 
        as a flat List of torch Tensors. This is private and serves as a utility 
        for self.flatten_distributions()
        """
        tensor_list: List[torch.Tensor] = []
        if self.continuous_tensor is not None:
            tensor_list.append(self.continuous_tensor)
        if self.all_discrete_list is not None:
            tensor_list.append(self.all_discrete_list)
        return tensor_list

    def flatten_distributions(self) -> torch.Tensor:
        """
        A utility method that returns all log probs in the ActionLogProbs in 
        a 2D tensor
        """
        return torch.cat(self._all_distributions_to_tensor_list(), dim=1)


