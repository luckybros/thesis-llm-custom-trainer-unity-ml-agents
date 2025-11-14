import torch
from typing import List, Optional, NamedTuple
from mlagents_plugin.utils.llm_action_log_probs import LLMActionLogProbs

discrete_list = torch.tensor([-5.0775e-03, -4.0332e-03, -5.9626e-03, -3.8591e-03])
all_discrete_list = torch.tensor([[-5.7662e+00, -6.2485e+00, -5.0775e-03],
        [-5.9974e+00, -6.4759e+00, -4.0332e-03],
        [-5.6380e+00, -6.0384e+00, -5.9626e-03],
        [-6.0812e+00, -6.4589e+00, -3.8591e-03]])

llm_action_log_probs = LLMActionLogProbs(
    continuous_tensor=None,
    discrete_list=discrete_list,
    all_discrete_list=all_discrete_list
)

result = llm_action_log_probs.flatten_distributions()

print(result)

