from mlagents_plugin.utils.llm_utils import LLMUtils
import numpy as np

ndarray_list = [
        [[-2.82471497, -1.34739019, -0.38455026]],    # lista non vuota
        [],                                           # lista vuota
        np.array([]),                                 # array vuoto
        np.array([[1.0, 2.0, 3.0]]),                  # array valido
        [[]],                                         # lista annidata vuota
    ]

clean = LLMUtils.clean_ndarray_list(ndarray_list)
print(clean)