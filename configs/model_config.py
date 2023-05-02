import torch.cuda
import torch.backends
import os

embedding_model_dict = {
    "ernie-tiny": "nghuyong/ernie-3.0-nano-zh",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec-base": "shibing624/text2vec-base-chinese",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

# supported LLM models
"""
llm_model_dict 处理了loader的一些预设行为，如加载位置，模型名称，模型处理器实例
"""
llm_model_dict = {
    "chatglm-6b-int4-qe": {
        "name": "chatglm-6b-int4-qe",
        "remote-checkpoint": "THUDM/chatglm-6b-int4-qe",
        "provides": "ChatGLM"
    },
    "chatglm-6b-int4": {
        "name": "chatglm-6b-int4",
        "remote-checkpoint": "THUDM/chatglm-6b-int4",
        "provides": "ChatGLM"
    },
    "chatglm-6b": {
        "name": "chatglm-6b",
        "remote-checkpoint": "THUDM/chatglm-6b-int4",
        "provides": "ChatGLM"
    },
    "llama-7b-hf": {
        "name": "llama-7b-hf",
        "remote-checkpoint": "llama-7b-hf",
        "provides": "LLamaLLM"
    },
    "vicuna-13b-hf": {
        "name": "vicuna-13b-hf",
        "remote-checkpoint": "vicuna-13b-hf",
        "provides": "LLamaLLM"
    },
    "chatyuan": {
        "name": "chatyuan",
        "remote-checkpoint": "ClueAI/ChatYuan-large-v2",
        "provides": None
    },
    "chatglm-6b-int8":{
        "name": "chatglm-6b-int8",
        "remote-checkpoint":  "THUDM/chatglm-6b-int8",
        "provides": "ChatGLM"
    },
}

# LLM model name
LLM_MODEL = "chatglm-6b"

# LLM streaming reponse
STREAMING = True

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = False

# remote in the model on loader checkpoint, if your load local model to add the ` --no-remote-model
NO_REMOTE_MODEL = False

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "vector_store", "")

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "content", "")

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 问题是：{question}"""

# 匹配后单段上下文长度
CHUNK_SIZE = 500