import logging
from typing import cast, Type, Union, Dict, Any

import numpy as np

from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.settings import TrainerSettings
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.buffer import BufferKey, ObservationKeyPrefix, RewardSignalUtil
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.optimizer import Optimizer
from mlagents.trainers.trainer.trainer_utils import get_gae
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic
from mlagents_plugin.trainers.torch_entities.networks import LLMSimpleActor, LLMSharedActorCritic
from mlagents_plugin.communicators.client.random_communication_client import RandomCommunicationClient
from mlagents_plugin.communicators.client.zmq_communicator_client import ZMQCommunicatorClient
from mlagents_plugin.trainers.settings import CommunicatorType

from mlagents_plugin.trainers.policy.llm_policy import TorchLLMPolicy
from mlagents_plugin.trainers.optimizer.torch_llm_optimizer import TorchLLMOptimizer, LLMSettings

logger = get_logger(__name__)

TRAINER_NAME = "llm_trainer"

class LLMTrainer(PPOTrainer):

    def __init__(
        self,
        behavior_name: str,
        reward_buff_cap: int,
        trainer_settings: TrainerSettings,
        training: bool,
        load: bool,
        seed: int,
        artifact_path: str,
    ):

        super().__init__(
            behavior_name,
            reward_buff_cap,
            trainer_settings,
            training,
            load, 
            seed,
            artifact_path
        )

        self.hyperparameters: LLMSettings = cast(
            LLMSettings, self.trainer_settings.hyperparameters
        )
        self.communicator_type = self.hyperparameters.communicator
        self.policy: TorchLLMPolicy = None

    def _process_trajectory(self, trajectory: Trajectory) -> None:
        """
        For now we copy and paste the logic of the parent class, because PPOTrainer
        append the buffer at the end of the function but we want to do all together
        with the LLM's log probs. 
        """
        # RL Trainer, we copy and paste bc if we called super()._process_trajectory it
        # would save all buffer without the LLM's log probs
        self._maybe_write_summary(self.get_step + len(trajectory.steps))
        self._maybe_save_model(self.get_step + len(trajectory.steps))
        self._increment_step(len(trajectory.steps), trajectory.behavior_id)

        agent_id = trajectory.agent_id  # All the agents should have the same ID

        #logger.info(f"agent_id: {agent_id}")
        agent_buffer_trajectory = trajectory.to_agentbuffer()

        # Check if we used group rewards, warn if so.
        self._warn_if_group_reward(agent_buffer_trajectory)

        # Update the normalization
        if self.is_training:
            self.policy.actor.update_normalization(agent_buffer_trajectory)
            self.optimizer.critic.update_normalization(agent_buffer_trajectory)

        (
            value_estimates,    
            value_next,
            value_memories,
        ) = self.optimizer.get_trajectory_value_estimates(
            agent_buffer_trajectory,
            trajectory.next_obs,
            trajectory.done_reached and not trajectory.interrupted,
        )
        if value_memories is not None:
            agent_buffer_trajectory[BufferKey.CRITIC_MEMORY].set(value_memories)

        # Aggiunge la lista di valori per segnale di ricompensa
        for name, v in value_estimates.items():
            agent_buffer_trajectory[RewardSignalUtil.value_estimates_key(name)].extend(
                v
            )
            self._stats_reporter.add_stat(
                f"Policy/{self.optimizer.reward_signals[name].name.capitalize()} Value Estimate",
                np.mean(v),
            )

        # Evaluate all reward functions
        self.collected_rewards["environment"][agent_id] += np.sum(
            agent_buffer_trajectory[BufferKey.ENVIRONMENT_REWARDS]
        )

        for name, reward_signal in self.optimizer.reward_signals.items():
            evaluate_result = (
                reward_signal.evaluate(agent_buffer_trajectory) * reward_signal.strength
            )
            agent_buffer_trajectory[RewardSignalUtil.rewards_key(name)].extend(
                evaluate_result
            )
            # Report the reward signals
            self.collected_rewards[name][agent_id] += np.sum(evaluate_result)

        # Compute GAE and returns
        tmp_advantages = []
        tmp_returns = []

        for name in self.optimizer.reward_signals:
            bootstrap_value = value_next[name]

            local_rewards = agent_buffer_trajectory[
                RewardSignalUtil.rewards_key(name)
            ].get_batch()

            local_value_estimates = agent_buffer_trajectory[
                RewardSignalUtil.value_estimates_key(name)
            ].get_batch()

            local_advantage = get_gae(
                rewards=local_rewards,
                value_estimates=local_value_estimates,
                value_next=bootstrap_value,
                gamma=self.optimizer.reward_signals[name].gamma,
                lambd=self.hyperparameters.lambd,
            )

            local_return = local_advantage + local_value_estimates
            # This is later use as target for the different value estimates
            agent_buffer_trajectory[RewardSignalUtil.returns_key(name)].set(
                local_return
            )
            agent_buffer_trajectory[RewardSignalUtil.advantage_key(name)].set(
                local_advantage
            )
            tmp_advantages.append(local_advantage)
            tmp_returns.append(local_return)

        # Get global advantages
        global_advantages = list(
            np.mean(np.array(tmp_advantages, dtype=np.float32), axis=0)
        )
        global_returns = list(np.mean(np.array(tmp_returns, dtype=np.float32), axis=0))
        agent_buffer_trajectory[BufferKey.ADVANTAGES].set(global_advantages)
        agent_buffer_trajectory[BufferKey.DISCOUNTED_RETURNS].set(global_returns)

        # le probs dell LLM sono a gruppi di time_horizon, voglio vedere
        # come le altre statistiche sono salvate
        # NEW: Adding LLM log probs
        num_steps = len(trajectory.steps)
        llm_log_probs = self.policy.pop_llm_buffer_data(agent_id=agent_id, num_items=num_steps)

        # è vero qui li raggruppo per timestamp ma solo perché sono stati salvati cosi prima
        # logger.info(f"llm_log_probs process_trajectory: {llm_log_probs}")
        #if not isinstance(llm_log_probs, dict):
        #    logger.warning(f"Attenzione: llm_log_probs non è un dict ma {type(llm_log_probs)}!")

        for key, item in llm_log_probs.items():
            agent_buffer_trajectory[key].extend(item)

        #logger.info(f"Trajectory: {agent_buffer_trajectory}")
        self._append_to_update_buffer(agent_buffer_trajectory)

        # logger.info(agent_buffer_trajectory)
        # If this was a terminal trajectory, append stats and reset reward collection
        if trajectory.done_reached:
            self._update_end_episode_stats(agent_id, self.optimizer)


    def create_optimizer(self) -> TorchLLMOptimizer:
        return TorchLLMOptimizer( # type: ignore
            cast(TorchLLMPolicy, self.policy), self.trainer_settings # type: ignore
        ) # type: ignore
    
    def create_policy(
            self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchLLMPolicy:
        
        actor_cls: Union[Type[LLMSimpleActor], Type[LLMSharedActorCritic]] = LLMSimpleActor
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }
        if self.shared_critic:
            reward_signal_configs = self.trainer_settings.reward_signals
            reward_signal_names = [
                key.value for key, _ in reward_signal_configs.items()
            ]
            actor_cls = LLMSharedActorCritic
            actor_kwargs.update({"stream_names": reward_signal_names})

        communicator_cls: Union[Type[RandomCommunicationClient], Type[ZMQCommunicatorClient]] = RandomCommunicationClient
        if self.communicator_type == CommunicatorType.RANDOM: 
            communicator_cls = RandomCommunicationClient
        if self.communicator_type == CommunicatorType.ZMQ:
            communicator_cls = ZMQCommunicatorClient
            
        policy = TorchLLMPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
            communicator_cls
        )
        
        return policy
    
    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

def get_type_and_setting():
    return {LLMTrainer.get_trainer_name(): LLMTrainer}, {
        LLMTrainer.get_trainer_name(): LLMSettings  # eventualmente riscrivere i settings per aggiungere iperparametro alpha
    }