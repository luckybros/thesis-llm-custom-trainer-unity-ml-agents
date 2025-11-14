from mlagents.trainers.buffer import BufferKey

class LLMBufferKey(BufferKey):
    LLM_CONTINUOUS_LOG_PROBS = "llm_continuous_log_probs"
    LLM_DISCRETE_LOG_PROBS = "llm_discrete_log_probs"