import torch.cuda
import torch.backends
from models.chatglm_llm import *
from models.llama_llm import LLamaLLM

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatglm-6b-int4-qe": {
        "path": "THUDM/chatglm-6b-int4-qe",
        "provides": ChatGLM
    },
    "chatglm-6b-int4": {
        "path": "THUDM/chatglm-6b-int4",
        "provides": ChatGLM
    },
    "chatglm-6b": {
        "path": "THUDM/chatglm-6b",
        "provides": ChatGLM
    },
    "llama-7b-hf": {
        "path": "llama-7b-hf",
        "provides": LLamaLLM
    },
}

# LLM model name
LLM_MODEL = "chatglm-6b"

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# remote in the model on loader checkpoint, if your load local model to add the ` --no-remote-model
NO_REMOTE_MODEL = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = "./vector_store/"

UPLOAD_ROOT_PATH = "./content/"

