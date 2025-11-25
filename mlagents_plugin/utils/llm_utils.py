import torch
import torch.nn.functional as F
from torch.distributions import Categorical

class LLMUtils: 

    @staticmethod
    def calculate_kl_distance(
        d1: list, 
        d2: list,
    ) -> torch.Tensor:
        """
        Evaluate KL distance between two distributions
        """
        # Per ora d1 dovrebbe essere una lista di tensori, dovrebbe esserlo
        # anche quella dell'LLM ma per ora aggiustiamo solo la prima        
        #print(f"d1: {d1}")
        #print(f"d2: {d2}")
        k_vals = []
        for logits1, logits2 in zip(d1, d2):
            logits1 = Categorical(logits=logits1)

            with torch.no_grad():
                logits2 = Categorical(logits=logits2)

            kl_loss = torch.distributions.kl_divergence(logits1, logits2).mean()
            k_vals.append(kl_loss)

        k_loss = torch.stack(k_vals).mean()
        return kl_loss
    
    @staticmethod
    def squeeze_list_dim(batch_list):
        """
        For single action, if you basically have only one action in reduces the 
        list on one dimention
        """
        if isinstance(batch_list, (list, tuple)) and len(batch_list) == 1 and isinstance(batch_list[0], (list, tuple)):
            return batch_list[0]
        return batch_list
    
    @staticmethod
    def tensor3d_to_list_of_2d(t: torch.Tensor) -> list:
        """
        For LLM distribution, we get them in a 3D tensor but they have to be
        in the same format of the log_probs, so a list of tensor, one for 
        action
        """
        t_transposed = t.permute(1, 0, 2)
        return list(torch.unbind(t_transposed, dim=0))
    