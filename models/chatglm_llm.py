
from abc import ABC

from langchain.llms.base import LLM
from typing import Optional, List
from models.loader import LoaderCheckPoint
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)

import transformers


class ChatGLM(BaseAnswer, LLM, ABC):
    max_token: int = 10000
    temperature: float = 0.01
    top_p = 0.9
    checkPoint: LoaderCheckPoint = None
    # history = []
    history_len: int = 10

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "ChatGLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None, ) -> str:
        pass

    def _generate_answer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False,
                         generate_with_callback: AnswerResultStream = None) -> None:
        # Create the StoppingCriteriaList with the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        stopping_criteria_list.append(listenerQueue)

        if streaming:

            for inum, (stream_resp, _) in enumerate(self.checkPoint.model.stream_chat(
                    self.checkPoint.tokenizer,
                    prompt,
                    history=history[-self.history_len:-1] if self.history_len > 0 else [],
                    max_length=self.max_token,
                    temperature=self.temperature,
                    stopping_criteria=stopping_criteria_list
            )):
                self.checkPoint.clear_torch_cache()
                if inum == 0:
                    history += [[prompt, stream_resp]]
                else:
                    history[-1] = [prompt, stream_resp]
                answer_result = AnswerResult()
                answer_result.history = history
                answer_result.llm_output = {"answer": stream_resp}
                answer_result.listenerToken = listenerQueue.listenerQueue.pop()
                generate_with_callback(answer_result)
        else:
            response, _ = self.checkPoint.model.chat(
                self.checkPoint.tokenizer,
                prompt,
                history=history[-self.history_len:] if self.history_len > 0 else [],
                max_length=self.max_token,
                temperature=self.temperature,
                stopping_criteria=stopping_criteria_list
            )
            self.checkPoint.clear_torch_cache()
            history += [[prompt, response]]
            answer_result = AnswerResult()
            answer_result.history = history
            answer_result.llm_output = {"answer": response}
            answer_result.listenerToken = listenerQueue.listenerQueue.pop()

            generate_with_callback(answer_result)


# if __name__ == "__main__":
    # 初始化消息
    # args = None
    # args = parser.parse_args()
    #
    # args_dict = vars(args)
    # loaderLLM = LoaderCheckPoint(args_dict)
    # llm = ChatGLM(loaderLLM)
    # llm.history_len = 10
    #
    # last_print_len = 0
    # for resp, history in llm._call("你好", streaming=True):
    #     print(resp[last_print_len:], end="", flush=True)
    #     last_print_len = len(resp)
    # for resp, history in llm._call("你好", streaming=False):
    #     print(resp)
    # pass
