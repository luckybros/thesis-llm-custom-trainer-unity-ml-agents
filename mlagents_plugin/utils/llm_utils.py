import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class LLMUtils: 

    @staticmethod
    def calculate_kl_distance(
        d1: torch.Tensor, 
        d2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Evaluate KL distance between two distributions
        """
        d1 = Categorical(logits=d1)

        with torch.no_grad():
            d2 = Categorical(logits=d2)

        kl_loss = torch.distributions.kl_divergence(d1, d2).mean()

        return kl_loss