import json
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens

from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
from configs.model_config import *
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from typing import Dict, Tuple, Union, Optional
from models.loader import LoaderLLM


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    llm: LoaderLLM = None
    # history = []
    tokenizer: object = None
    model: object = None
    history_len: int = 10
    callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

    def __init__(self, llm: LoaderLLM = None):
        super().__init__()
        self.llm = llm

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    def _call(self,
              prompt: str,
              history: List[List[str]] = [],
              streaming: bool = STREAMING):  # -> Tuple[str, List[List[str]]]:
        if streaming:
            for inum, (stream_resp, _) in enumerate(self.model.stream_chat(
                    self.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    temperature=self.temperature,
            )):
                torch_gc(DEVICE)
                if inum == 0:
                    history += [[prompt, stream_resp]]
                else:
                    history[-1] = [prompt, stream_resp]
                yield stream_resp, history
        else:
            response, _ = self.model.chat(
                    self.tokenizer,
                    prompt,
                    history=history[-self.history_len:] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    temperature=self.temperature,
            )
            torch_gc(DEVICE)
            history += [[prompt, response]]
            yield response, history

    # def chat(self,
    #          prompt: str) -> str:
    #     response, _ = self.model.chat(
    #         self.tokenizer,
    #         prompt,
    #         history=self.history[-self.history_len:] if self.history_len > 0 else [],
    #         max_length=self.max_token,
    #         temperature=self.temperature,
    #     )
    #     torch_gc()
    #     self.history = self.history + [[None, response]]
    #     return response



if __name__ == "__main__":
    llm = ChatGLM()
    llm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL],
                   llm_device=LLM_DEVICE, )
    last_print_len=0
    for resp, history in llm._call("你好", streaming=True):
        print(resp[last_print_len:], end="", flush=True)
        last_print_len = len(resp)
    for resp, history in llm._call("你好", streaming=False):
        print(resp)
    pass
