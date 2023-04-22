import json
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens

from typing import Dict, Tuple, Union, Optional
from models.loader import LoaderLLM


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    history = []
    llm: LoaderLLM = None
    history_len: int = 10

    def __init__(self, llm: LoaderLLM = None):
        super().__init__()
        self.llm = llm

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        response, _ = self.llm.model.chat(
            self.llm.tokenizer,
            prompt,
            history=self.history[-self.history_len:] if self.history_len > 0 else [],
            max_length=self.max_token,
            temperature=self.temperature,
        )
        self.llm.clear_torch_cache()
        if stop is not None:
            response = enforce_stop_tokens(response, stop)

        self.history = self.history + [[None, response]]
        return response
