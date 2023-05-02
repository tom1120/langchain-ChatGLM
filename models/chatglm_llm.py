import json
from langchain.llms.base import LLM
from typing import Optional, List
from langchain.llms.utils import enforce_stop_tokens

from models.loader.args import parser
from configs.model_config import *
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from models.loader import LoaderCheckPoint


class ChatGLM(LLM):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    llm: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, llm: LoaderCheckPoint = None):
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
            for inum, (stream_resp, _) in enumerate(self.llm.model.stream_chat(
                    self.llm.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    temperature=self.temperature,
            )):
                self.llm.clear_torch_cache()
                if inum == 0:
                    history += [[prompt, stream_resp]]
                else:
                    history[-1] = [prompt, stream_resp]
                yield stream_resp, history
        else:
            response, _ = self.llm.model.chat(
                self.llm.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
                temperature=self.temperature,
            )
            self.llm.clear_torch_cache()
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
    # 初始化消息
    args = None
    args = parser.parse_args()

    args_dict = vars(args)
    loaderLLM = LoaderCheckPoint(args_dict)
    llm = ChatGLM(loaderLLM)
    llm.history_len = 10

    last_print_len = 0
    for resp, history in llm._call("你好", streaming=True):
        print(resp[last_print_len:], end="", flush=True)
        last_print_len = len(resp)
    for resp, history in llm._call("你好", streaming=False):
        print(resp)
    pass
