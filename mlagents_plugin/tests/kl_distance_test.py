import torch
from mlagents_plugin.utils.llm_utils import LLMUtils

d1 = torch.tensor([[-5.7662e+00, -6.2485e+00, -5.0775e-03],
        [-5.9974e+00, -6.4759e+00, -4.0332e-03],
        [-5.6380e+00, -6.0384e+00, -5.9626e-03],
        [-6.0812e+00, -6.4589e+00, -3.8591e-03]])
d2 = torch.tensor([[-5.7662e+00, -6.2485e+00, -5.0775e-03],
        [-5.9974e+00, -6.4759e+00, -4.0332e-03],
        [-5.6380e+00, -6.0384e+00, -5.9626e-03],
        [-6.0812e+00, -6.4589e+00, -3.8591e-03]])

print(LLMUtils.calculate_kl_distance(d1, d2))

