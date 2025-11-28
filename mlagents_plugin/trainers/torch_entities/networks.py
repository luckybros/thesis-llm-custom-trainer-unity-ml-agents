from typing import Any, Dict, List, Optional

import torch
from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic
from mlagents_envs.base_env import ActionSpec, ObservationSpec
from mlagents_plugin.trainers.torch_entities.action_model import LLMActionModel
from mlagents.trainers.settings import NetworkSettings
from mlagents.trainers.torch_entities.agent_action import AgentAction

class LLMSimpleActor(SimpleActor):
    """
    Simple actor that extends action decision giving information about
    continuous action choices to the policy
    """
    def __init__(
        self,
        observation_specs: List[ObservationSpec],
        network_settings: NetworkSettings,
        action_spec: ActionSpec,
        conditional_sigma: bool = False,
        tanh_squash: bool = False,
    ):
        super().__init__(observation_specs, network_settings, action_spec, conditional_sigma, tanh_squash)

        self.action_model = LLMActionModel(
            self.encoding_size, 
            action_spec,
            conditional_sigma=conditional_sigma,
            tanh_squash=tanh_squash, 
            deterministic=network_settings.deterministic,
        )

    def get_stats(
        self,
        inputs: List[torch.Tensor],
        actions: AgentAction,
        masks: Optional[torch.Tensor] = None,
        memories: Optional[torch.Tensor] = None,
        sequence_length: int = 1,
    ) -> Dict[str, Any]:
        encoding, actor_mem_outs = self.network_body(
            inputs, memories=memories, sequence_length=sequence_length
        )

        log_probs, entropies, continuous_parameters = self.action_model.evaluate(encoding, masks, actions)

        run_out = {}
        run_out["log_probs"] = log_probs
        run_out["entropy"] = entropies
        run_out["continuous_parameters"] = None
        if continuous_parameters is not None:
            run_out["continuous_parameters"] = continuous_parameters
        return run_out


class LLMSharedActorCritic(SharedActorCritic):
    """
    Simple actor that extends action decision giving information about
    continuous action choices to the policy
    """
    pass