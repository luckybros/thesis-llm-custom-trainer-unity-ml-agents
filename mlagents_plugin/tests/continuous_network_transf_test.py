import torch
from mlagents_plugin.utils.llm_utils import LLMUtils

import torch

continuous_parameters = {
    "mean": torch.tensor([[1.1, 2.2], [3.3, 4.4]]),  # shape [N, A]
    "std": torch.tensor([[0.1, 0.2], [0.3, 0.4]])    # shape [N, A]
}

print(LLMUtils.continuous_net_parameters_transform(continuous_parameters))

