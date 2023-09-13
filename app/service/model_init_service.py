from chains.local_doc_qa import LocalDocQA
from configs.model_config import LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DEVICE, LLM_HISTORY_LEN, VECTOR_SEARCH_TOP_K

local_doc_qa = LocalDocQA()
local_doc_qa.init_cfg(
    llm_model=LLM_MODEL,
    embedding_model=EMBEDDING_MODEL,
    embedding_device=EMBEDDING_DEVICE,
    llm_history_len=LLM_HISTORY_LEN,
    top_k=VECTOR_SEARCH_TOP_K,
)