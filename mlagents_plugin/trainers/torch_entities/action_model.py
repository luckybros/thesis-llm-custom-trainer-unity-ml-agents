from typing import Dict, Tuple
import torch
from mlagents.trainers.torch_entities.action_model import ActionModel
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents_envs.logging_util import get_logger

logger = get_logger(__name__)

class LLMActionModel(ActionModel):
    def evaluate(
        self, inputs: torch.Tensor, masks: torch.Tensor, actions: AgentAction
    ) -> Tuple[ActionLogProbs, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Given actions and encoding from the network body, gets the distributions and
        computes the log probabilites and entropies.
        :params inputs: The encoding from the network body
        :params masks: Action masks for discrete actions
        :params actions: The AgentAction
        :return: An ActionLogProbs tuple and a torch tensor of the distribution entropies.
        """
        dists = self._get_dists(inputs, masks) # instance of GaussianDistInstance
        #logger.info(f"conditional_sigma: {self.conditional_sigma}")
        #logger.info(f"mean: {dists.continuous.mean}, std: {dists.continuous.std}")
        log_probs, entropies = self._get_probs_and_entropy(actions, dists)
        # Use the sum of entropy across actions, not the mean
        entropy_sum = torch.sum(entropies, dim=1)

        params_dict = None
        if dists.continuous is not None:
            params_dict = {}
            params_dict["mean"] = dists.continuous.mean
            params_dict["std"] = dists.continuous.std
        
        return log_probs, entropy_sum, params_dict