from mlagents.trainers.ghost.trainer import GhostTrainer
from mlagents.trainers.policy import Policy
from mlagents.trainers.behavior_id_utils import (
    BehaviorIdentifiers,
    create_name_behavior_id,
)
from mlagents_envs.base_env import BehaviorSpec
from mlagents_envs.logging_util import get_logger
from mlagents_plugin.trainers.policy.llm_policy import TorchLLMPolicy

logger = get_logger(__name__)

class LLMGhostTrainer(GhostTrainer):
    
    def create_policy(
        self,
        parsed_behavior_id: BehaviorIdentifiers,
        behavior_spec: BehaviorSpec,
    ) -> Policy:
        policy = super().create_policy(parsed_behavior_id, 
            behavior_spec)
        team_id = parsed_behavior_id.team_id

        # freeze policy if it's not in the training team
        if (isinstance(policy, TorchLLMPolicy)):
            if team_id != self.wrapped_trainer_team:
                policy.set_ghost_frozen(True)
            else:
                policy.set_ghost_frozen(False)
        return policy
    
    def _swap_snapshots(self) -> None:
        super()._swap_snapshots()
        self._update_all_frozen_flags()

    def _update_all_frozen_flags(self) -> None:
        for team_id in self._team_to_name_to_policy_queue:
            for brain_name in self._team_to_name_to_policy_queue[team_id]:
                behavior_id = create_name_behavior_id(brain_name, team_id)
                policy = self.get_policy(behavior_id)
                if isinstance(policy, TorchLLMPolicy):
                    is_frozen = (team_id != self._learning_team)
                    policy.set_ghost_frozen(is_frozen)
                    
                    # IL PASSAGGIO CRITICO:
                    # Rimettiamo la policy nella coda in modo che l'Environment Worker 
                    # riceva l'oggetto aggiornato con il nuovo valore del flag!
                    self._team_to_name_to_policy_queue[team_id][brain_name].put(policy)

    