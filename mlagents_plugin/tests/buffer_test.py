# [azione][timestamp][dist]
"""a = [[[0.3, 0.2, 0.5], 
      [0.3, 0.2, 0.5]], 
      [[0.2, 0.8],
       [0.2, 0.8]]]

# proviamo ad appendere una nuova distribuzione alla prima azione, indice 0 quindi
a[0].append([0.5, 0.2, 0.3])

print(a)"""

from mlagents_plugin.trainers.llm_buffer import LLMBuffer, LLMBufferKey

buffer = LLMBuffer()
buffer.add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, [[0.3, 0.6, 0.1], [0.2, 0.8]])
buffer.add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, [[0.4, 0.4, 0.2], [0.1, 0.9]])
buffer.add_entry(LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS, [[0.4, 0.4, 0.2], [0.1, 0.9]])
# Restituisce la chiave e il valore, il dizionario quindi
entry = buffer.pop_n_entries(3)
print(len(entry[LLMBufferKey.LLM_LOG_DISCRETE_LOG_PROBS]))

