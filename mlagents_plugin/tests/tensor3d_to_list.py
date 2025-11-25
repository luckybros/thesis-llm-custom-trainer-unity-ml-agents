import torch
from mlagents_plugin.utils.llm_utils import LLMUtils
t = torch.randn(5, 3, 3)   # 5 timestamp, 2 azioni, 3 opzioni per azione

print(f"Before: {t}")
print(f"After: {LLMUtils.tensor3d_to_list_of_2d(t)}")


