import logging
from typing import Any, Dict, Type, Union

from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.settings import TrainerSettings
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.optimizer import Optimizer
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.ppo.optimizer_torch import PPOSettings
from mlagents.trainers.ppo.trainer import PPOTrainer
from mlagents.trainers.torch_entities.networks import SimpleActor, SharedActorCritic

from .llm_policy import TorchLLMPolicy

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

        logger.info("**************************************************")
        logger.info(f"***** PLUGIN CARICATO! (Metodo Esplicito) *****")
        logger.info("**************************************************")

        self.policy: TorchLLMPolicy = None

    def create_optimizer(self) -> TorchLLMOptimizer:
        return TorchLLMOptimizer( # type: ignore
            cast(TorchLLMPolicy, self.policy), self.trainer_settings # type: ignore
        ) # type: ignore
    
    def create_policy(
            self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec
    ) -> TorchLLMPolicy:
        
        actor_cls: Union[Type[SimpleActor], Type[SharedActorCritic]] = SimpleActor
        actor_kwargs: Dict[str, Any] = {
            "conditional_sigma": False,
            "tanh_squash": False,
        }
        if self.shared_critic:
            reward_signal_configs = self.trainer_settings.reward_signals
            reward_signal_names = [
                key.value for key, _ in reward_signal_configs.items()
            ]
            actor_cls = SharedActorCritic
            actor_kwargs.update({"stream_names": reward_signal_names})

        policy = TorchLLMPolicy(
            self.seed,
            behavior_spec,
            self.trainer_settings.network_settings,
            actor_cls,
            actor_kwargs,
        )
        
        return policy
    
    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

def get_type_and_setting():
    return {LLMTrainer.get_trainer_name(): LLMTrainer}, {
        LLMTrainer.get_trainer_name(): PPOSettings  # eventualmente riscrivere i settings per aggiungere iperparametro alpha
    }