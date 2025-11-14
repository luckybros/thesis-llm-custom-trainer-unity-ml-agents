import logging

from mlagents.trainers.trainer.rl_trainer import RLTrainer
from mlagents.trainers.settings import TrainerSettings
from mlagents_envs.base_env import BehaviorSpec
from mlagents.trainers.behavior_id_utils import BehaviorIdentifiers
from mlagents.trainers.trajectory import Trajectory
from mlagents.trainers.optimizer import Optimizer
from mlagents_envs.logging_util import get_logger
from mlagents.trainers.settings import TrainerSettings
from mlagents.trainers.ppo.optimizer_torch import PPOSettings

from .llm_policy import LLMPolicy

logger = get_logger(__name__)

TRAINER_NAME = "llm_trainer"

class LLMTrainer(RLTrainer):

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
            trainer_settings,
            training,
            load, 
            artifact_path,
            reward_buff_cap
        )

        logger.info("**************************************************")
        logger.info(f"***** PLUGIN CARICATO! (Metodo Esplicito) *****")
        logger.info("**************************************************")

        self.policy: LLMPolicy = None
        self.seed = seed
    
    def create_policy(self, parsed_behavior_id: BehaviorIdentifiers, behavior_spec: BehaviorSpec) -> LLMPolicy:
        
        policy = LLMPolicy(
            seed = self.seed,
            behavior_spec=behavior_spec,
            network_settings=self.trainer_settings.network_settings
        )

        return policy
    
    def add_policy(
        self, parsed_behavior_id: BehaviorIdentifiers, policy: LLMPolicy
    ): 
        """
        Uno dei metodi astratti necessari da implementare per il trainer, è implementato in on_policy_trainer.py
        """
        if self.policy:
            logger.warning(
                "Your environment contains multiple teams, but {} doesn't support adversarial games. Enable self-play to \
                    train adversarial games.".format(
                    self.__class__.__name__
                )
            )
        self.policy = policy
        self.policies[parsed_behavior_id.behavior_id] = policy
        self._step = policy.get_current_step()
        logger.info(f"Policy '{policy.__class__.__name__}' aggiunta al trainer '{TRAINER_NAME}'.")

    def _process_trajectory(self, trajectory : Trajectory) -> None:
        pass

    def create_optimizer(self) -> Optimizer:
        """
        Metodo astratto richiesto dalla classe base RLTrainer.
        Poiché il nostro trainer non si allena, non abbiamo un ottimizzatore.
        Restituiamo None per soddisfare il contratto della classe base.
        """
        logger.info("create_optimizer() chiamato, ma non è necessario per LLMTrainer. Restituisco None.")
        return None

    def _update_policy(self) -> bool:
        """
        Questo metodo è richiesto da RLTrainer.
        Lo implementiamo ma restituiamo False per segnalare che non è
        avvenuto nessun aggiornamento del modello.
        """
        return False
    
    def _is_ready_update(self) -> bool:
        """
        Uno dei metodi astratti necessari da implementare per il trainer, è implementato in on_policy_trainer.py
        """
        return False

    @staticmethod
    def get_trainer_name() -> str:
        return TRAINER_NAME

def get_type_and_setting():
    return {LLMTrainer.get_trainer_name(): LLMTrainer}, {
        LLMTrainer.get_trainer_name(): PPOSettings
    }