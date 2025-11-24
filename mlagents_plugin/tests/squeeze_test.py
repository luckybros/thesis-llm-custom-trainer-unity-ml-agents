import torch
from mlagents_plugin.utils.llm_utils import LLMUtils

dist = [[-0.8430481982192435, -1.2475810744458697, -1.2644139791534528]]

print(f"dist before: {dist}")
print(f"dist after: {LLMUtils.squeeze_list_dim(dist)}")