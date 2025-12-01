from typing import Dict, cast

import attr
from mlagents.trainers.ppo.optimizer_torch import TorchPPOOptimizer, PPOSettings
from mlagents.trainers.settings import (
    TrainerSettings,
    OnPolicyHyperparamSettings,
    ScheduleType,
)
from mlagents.trainers.torch_entities.utils import ModelUtils
from mlagents_plugin.trainers.llm_buffer import LLMBufferKey
from mlagents.trainers.torch_entities.agent_action import AgentAction
from mlagents.trainers.buffer import AgentBuffer, BufferKey, RewardSignalUtil
from mlagents.trainers.trajectory import ObsUtil
from mlagents_envs.timers import timed
from mlagents.torch_utils import torch, default_device
from mlagents.trainers.torch_entities.action_log_probs import ActionLogProbs
from mlagents_plugin.utils.llm_utils import LLMUtils
from mlagents_plugin.trainers.policy.llm_policy import TorchLLMPolicy
from mlagents_envs.logging_util import get_logger
from mlagents_plugin.trainers.settings import CommunicatorType


logger = get_logger(__name__)

@attr.s(auto_attribs=True)
class LLMSettings(PPOSettings):
    alpha : float = 0.1
    communicator : CommunicatorType = CommunicatorType.RANDOM
    
class TorchLLMOptimizer(TorchPPOOptimizer):
    def __init__(self, policy: TorchLLMPolicy, trainer_settings: TrainerSettings):
        super().__init__(policy, trainer_settings)
        self.hyperparameters: LLMSettings = cast(
            LLMSettings, trainer_settings.hyperparameters
        )
        self.alpha = self.hyperparameters.alpha
        """
        Eventualmente,
        per alpha relativo alla distanza,
        self.decay_alpha = ModelUtils.DecayedValue(
            self.hyperparameters.alpha_schedule,
            self.hyperparameters.alpha,
            1e-5,
            self.trainer_settings.max_steps,
        )
        """

    @timed
    def update(self, batch: AgentBuffer, num_sequences: int) -> Dict[str, float]:

        decay_lr = self.decay_learning_rate.get_value(self.policy.get_current_step())
        decay_eps = self.decay_epsilon.get_value(self.policy.get_current_step())
        decay_bet = self.decay_beta.get_value(self.policy.get_current_step())
        decay_alpha = self.alpha    # per ora valore fisso

        returns = {}   
        old_values = {}   
        for name in self.reward_signals:
            old_values[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.value_estimates_key(name)]
            )
            returns[name] = ModelUtils.list_to_tensor(
                batch[RewardSignalUtil.returns_key(name)]
            )

        n_obs = len(self.policy.behavior_spec.observation_specs)    
        current_obs = ObsUtil.from_buffer(batch, n_obs)    
        current_obs = [ModelUtils.list_to_tensor(obs) for obs in current_obs]
        
        act_masks = ModelUtils.list_to_tensor(batch[BufferKey.ACTION_MASK]) 
        actions = AgentAction.from_buffer(batch)
        
        memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.MEMORY][i])
            for i in range(0, len(batch[BufferKey.MEMORY]), self.policy.sequence_length)
        ]
        if len(memories) > 0:
            memories = torch.stack(memories).unsqueeze(0)

        # Get value memories
        value_memories = [
            ModelUtils.list_to_tensor(batch[BufferKey.CRITIC_MEMORY][i])
            for i in range(
                0, len(batch[BufferKey.CRITIC_MEMORY]), self.policy.sequence_length
            )
        ]
        if len(value_memories) > 0:
            value_memories = torch.stack(value_memories).unsqueeze(0)

        run_out = self.policy.actor.get_stats(
            current_obs,
            actions,
            masks=act_masks,
            memories=memories,
            sequence_length=self.policy.sequence_length,
        )

        log_probs = run_out["log_probs"]
        entropy = run_out["entropy"]
        continuous_log_probs = run_out["continuous_parameters"]

        # Net output
        #logger.info(f"continuous_parameters: {continuous_log_probs}")

        values, _ = self.critic.critic_pass(
            current_obs,
            memories=value_memories,
            sequence_length=self.policy.sequence_length,
        )

        llm_discrete_log_probs = None
        llm_continuous_log_probs = None
        
        # Eventually write a static method of a class that does that
        if LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS in batch:
            llm_discrete_log_probs = ModelUtils.list_to_tensor(
                batch[LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS]
            )
            llm_discrete_log_probs = LLMUtils.tensor3d_to_list_of_2d(llm_discrete_log_probs)

        if LLMBufferKey.LLM_LOG_CONTINUOUS_LOG_PROBS in batch:
            #logger.info(f"llm_continuous_log_probs pre : {llm_continuous_log_probs}")
            llm_continuous_log_probs = ModelUtils.list_to_tensor(
                batch[LLMBufferKey.LLM_LOG_CONTINUOUS_LOG_PROBS]
            )
            llm_continuous_log_probs = LLMUtils.tensor3d_to_list_of_2d(llm_continuous_log_probs)
            #logger.info(f"llm_continuous_log_probs after : {llm_continuous_log_probs}")
            
        # At this point, llm_discrete_log_probs is a 3D tensor with this dimentions: [timestamp][action_type][action_dist]
        # We should it became a list of 2d tensor on the actions.
        #logger.info(f"llm_discrete_log_probs pre: {llm_discrete_log_probs}")
        #logger.info(f"llm_discrete_log_probs pre: {llm_discrete_log_probs}")

        old_log_probs = ActionLogProbs.from_buffer(batch)
        
        log_probs_action = log_probs.flatten()
        old_log_probs_action = old_log_probs.flatten()
        
        loss_masks = ModelUtils.list_to_tensor(batch[BufferKey.MASKS], dtype=torch.bool)

        value_loss = ModelUtils.trust_region_value_loss(
            values, old_values, returns, decay_eps, loss_masks
        )

        policy_loss = ModelUtils.trust_region_policy_loss(
            ModelUtils.list_to_tensor(batch[BufferKey.ADVANTAGES]),
            log_probs_action,
            old_log_probs_action,
            loss_masks,
            decay_eps,
        )
        
        # all_discrete_list restituisce una lista di tensori, un tensore per azione!
        # devo fare lo stesso per l'LLM che attualmente non raggruppa tutte le prob per azione 
        discrete_log_probs = log_probs.all_discrete_list
        if continuous_log_probs is not None:
            continuous_log_probs = LLMUtils.continuous_net_parameters_transform(continuous_log_probs)
        #logger.info(f"current_obs: {current_obs}")
        #logger.info(f"action: {actions}")
        #logger.info(f"log_probs in Optimizer: {log_probs}")
        #logger.info(f"llm_discrete_log_probs in Optimizer: {llm_discrete_log_probs}")
        #assert log_probs[0].shape == llm_discrete_log_probs[0].shape
        llm_loss = LLMUtils.calculate_kl_distance(discrete_log_probs, llm_discrete_log_probs, continuous_log_probs, llm_continuous_log_probs)

        #logger.info(f"LLM Loss: {llm_loss}")
        loss = (
            policy_loss
            + 0.5 * value_loss
            + decay_alpha * llm_loss
            - decay_bet * ModelUtils.masked_mean(entropy, loss_masks)
        )

        #logger.info(f"Loss: {loss}")
        # Set optimizer learning rate
        ModelUtils.update_learning_rate(self.optimizer, decay_lr)

        self.optimizer.zero_grad()

        loss.backward()

        self.optimizer.step()
        update_stats = {
            # NOTE: abs() is not technically correct, but matches the behavior in TensorFlow.
            # TODO: After PyTorch is default, change to something more correct.
            "Losses/Policy Loss": torch.abs(policy_loss).item(),
            "Losses/Value Loss": value_loss.item(),
            "Policy/Learning Rate": decay_lr,
            "Policy/Epsilon": decay_eps,
            "Policy/Beta": decay_bet,
        }

        return update_stats