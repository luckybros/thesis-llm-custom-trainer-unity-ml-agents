from typing import List, Union
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical, Normal, kl_divergence

class LLMUtils: 

    @staticmethod
    def calculate_kl_distance(
        discrete_1: list, 
        discrete_2: list,
        continuous_1: list,
        continuous_2: list,
    ) -> torch.Tensor:
        """
        Evaluate KL distance between two distributions
        """
        # Per ora d1 dovrebbe essere una lista di tensori, dovrebbe esserlo
        # anche quella dell'LLM ma per ora aggiustiamo solo la prima        
        k_vals = []
        if discrete_1 is not None and discrete_2 is not None:
            for logits1, logits2 in zip(discrete_1, discrete_2):
                cat_logits1 = Categorical(logits=logits1)

                with torch.no_grad():
                    eps = 1e-6
                    smoothed_probs = logits2 + eps
                    # Rinormalizziamo per far sì che la somma sia esattamente 1.0
                    smoothed_probs = smoothed_probs / smoothed_probs.sum(dim=-1, keepdim=True)
                    cat_logits2 = Categorical(probs=smoothed_probs)

                kl_loss = kl_divergence(cat_logits1, cat_logits2).mean()
                k_vals.append(kl_loss)

        if continuous_1 is not None and continuous_2 is not None:
            for logits1, logits2 in zip(continuous_1, continuous_2):
                #print(f"logits1: {logits1}, logits2 {logits2}")
                means_first = logits1[:, 0]
                #print(f"means_first: {means_first}")
                stds_first = logits1[:, 1]
                #print(f"stds_first: {stds_first}")
                means_second = logits2[:, 0]
                #print(f"means_second: {means_second}")
                stds_second = logits2[:, 1]
                #print(f"stds_second: {stds_second}")
                dist1 = Normal(means_first, stds_first)

                with torch.no_grad():
                    dist2 = Normal(means_second, stds_second)

                kl_loss = kl_divergence(dist1, dist2).mean()
                k_vals.append(kl_loss)
                
        k_loss = torch.stack(k_vals).mean()
        return k_loss
    
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
    
    @staticmethod
    def continuous_net_parameters_transform(continuous_parameters) -> list:
        means = continuous_parameters["mean"]
        stds = continuous_parameters["std"]
        out = []
        for a in range(means.shape[1]):
            mean_col = means[:, a].unsqueeze(1)
            std_col = stds[:, a].unsqueeze(1)
            out.append(torch.cat([mean_col, std_col], dim=1))
        return out
    
    @staticmethod
    def clean_ndarray_list(ndarray_list):
        """
        Restituisce una nuova lista composta solo dagli item NON vuoti:
        - Elimina item che sono liste vuote []
        - Elimina item che sono array numpy di shape (0,)
        """
        out = []
        for elem in ndarray_list:
            # Se è una lista vuota
            if isinstance(elem, list) and len(elem) == 0:
                continue
            # Se è un array numpy vuoto
            if isinstance(elem, np.ndarray) and elem.size == 0:
                continue
            # Se è una lista annidata, escludi anche [ [] ]
            if isinstance(elem, list) and len(elem) == 1 and isinstance(elem[0], list) and len(elem[0]) == 0:
                continue
            out.append(elem)
        return out
    
    @staticmethod 
    def filter_log_probs(discrete_log_probs: List[torch.Tensor], mask: Union[List[int], torch.Tensor]) -> List[torch.Tensor]:
        """
        Prende una lista di tensori [T, num_branches] (uno per azione)
        e una mask binaria (lunghezza T) e ritorna la lista di tensori filtrati solo sugli step dove mask==1
        """
        # Assicurati che la mask sia torch tensor booleano
        mask_tensor = torch.as_tensor(mask).bool()
        filtered = [
            x[mask_tensor] for x in discrete_log_probs
        ]
        return filtered
    
    @staticmethod
    def transpose_llm_probs_to_mlagents_format(llm_probs: list) -> list:
        """
        Trasforma una lista 3D [timestamp][n_azioni][distribuzione]
        in una lista di tensori 2D, uno per ogni azione [timestamp][distribuzione].
        """
        # Controllo di sicurezza: se la lista è vuota, restituiamo una lista vuota
        if not llm_probs or not llm_probs[0]:
            return []

        # llm_probs[0] contiene le azioni del primo timestamp
        num_actions = len(llm_probs[0]) 
        tensor_list = []

        # Iteriamo su ogni singola azione (es: 0=Move, 1=Turn, 2=Shoot)
        for a in range(num_actions):
            # Estraiamo i dati dell'azione 'a' per TUTTI i timestamp
            # List comprehension che raggruppa verticalmente
            action_data_across_time = [timestep[a] for timestep in llm_probs]
            
            # Ora abbiamo un rettangolo perfetto, possiamo farne un tensore!
            action_tensor = torch.as_tensor(action_data_across_time, dtype=torch.float32)
            tensor_list.append(action_tensor)

        return tensor_list