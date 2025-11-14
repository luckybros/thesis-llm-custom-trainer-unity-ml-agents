import torch
from mlagents_plugin.utils.llm_utils import LLMUtils

d1 = torch.tensor([[0.0031, 0.0019, 0.9949],
        [0.0025, 0.0015, 0.9960],
        [0.0036, 0.0024, 0.9941],
        [0.0023, 0.0016, 0.9961]])
d2 = torch.tensor([[0.0031, 0.0019, 0.9949],
        [0.0025, 0.0015, 0.9960],
        [0.0036, 0.0024, 0.9941],
        [0.0023, 0.0016, 0.9961]])

print(LLMUtils.calculate_kl_distance(d1, d2))

