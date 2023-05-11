from abc import ABC

from langchain.llms.base import LLM

import torch
import transformers
from typing import Optional, List, Dict, Any
from models.loader import LoaderCheckPoint
from models.extensions.callback import (Iteratorize, Stream, FixedLengthQueue)
import models.shared as shared
from models.base import (BaseAnswer,
                         AnswerResult,
                         AnswerResultStream,
                         AnswerResultQueueSentinelTokenListenerQueue)
from langchain.callbacks.manager import (
    CallbackManagerForLLMRun
)


def _streaming_response_template() -> Dict[str, Any]:
    """
    :return: 响应结构
    """
    return {
        "text": ""
    }


def _update_response(response: Dict[str, Any], stream_response: str) -> None:
    """Update response from the stream response."""
    response["text"] += stream_response


class LLamaLLM(BaseAnswer, LLM, ABC):
    checkPoint: LoaderCheckPoint = None
    history = []
    history_len: int = 10

    generate_params: object = {'max_new_tokens': 50,
                               'do_sample': True,
                               'temperature': 0.7,
                               'top_p': 0.1,
                               'typical_p': 1,
                               'repetition_penalty': 1.18,
                               'encoder_repetition_penalty': 1,
                               'top_k': 40, 'min_length': 0,
                               'no_repeat_ngram_size': 0,
                               'num_beams': 1,
                               'penalty_alpha': 0,
                               'length_penalty': 1,
                               'early_stopping': False,
                               'eos_token_id': [2],
                               'stopping_criteria': []
                               }
    state: object = {'max_new_tokens': 50,
                     'seed': 1,
                     'temperature': 0, 'top_p': 0.1,
                     'top_k': 40, 'typical_p': 1,
                     'repetition_penalty': 1.18,
                     'encoder_repetition_penalty': 1,
                     'no_repeat_ngram_size': 0,
                     'min_length': 0,
                     'do_sample': True,
                     'penalty_alpha': 0,
                     'num_beams': 1,
                     'length_penalty': 1,
                     'early_stopping': False, 'add_bos_token': True, 'ban_eos_token': False,
                     'truncation_length': 2048, 'custom_stopping_strings': '',
                     'cpu_memory': 0, 'auto_devices': False, 'disk': False, 'cpu': False, 'bf16': False,
                     'load_in_8bit': False, 'wbits': 'None', 'groupsize': 'None', 'model_type': 'None',
                     'pre_layer': 0, 'gpu_memory_0': 0}

    def __init__(self, checkPoint: LoaderCheckPoint = None):
        super().__init__()
        self.checkPoint = checkPoint

    @property
    def _llm_type(self) -> str:
        return "LLamaLLM"

    @property
    def _check_point(self) -> LoaderCheckPoint:
        return self.checkPoint

    def encode(self, prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        input_ids = self.checkPoint.tokenizer.encode(str(prompt), return_tensors='pt',
                                              add_special_tokens=add_special_tokens)
        # This is a hack for making replies more creative.
        if not add_bos_token and input_ids[0][0] == self.checkPoint.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(self.checkPoint.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        return input_ids.cuda()

    def decode(self, output_ids):
        reply = self.checkPoint.tokenizer.decode(output_ids, skip_special_tokens=True)
        reply = reply.replace(r'<|endoftext|>', '')
        return reply

    def get_max_prompt_length(self):
        max_length = self.state['truncation_length'] - self.state['max_new_tokens']
        return max_length

    def generate_with_callback(self, callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        self.checkPoint.clear_torch_cache()
        with torch.no_grad():
            self.checkPoint.model.generate(**kwargs)

    def generate_with_streaming(self, callback=None, **kwargs):
        return Iteratorize(self.generate_with_callback, kwargs, callback)

    # 将历史对话数组转换为文本格式
    def history_to_text(self):
        formatted_history = ''
        history = self.history[-self.history_len:] if self.history_len > 0 else []
        for entry in history:
            role, content = entry
            formatted_history += f"### {role}: {content}\n"
        return formatted_history

    def generate_softprompt_history_tensors(self, input_ids):
        """
        历史对话软提示
            这段代码首先定义了一个名为 history_to_text 的函数，用于将 self.history
            数组转换为所需的文本格式。然后，我们将格式化后的历史文本
            再用 self.encode 将其转换为向量表示。最后，将历史对话向量与当前输入的对话向量拼接在一起。
        :return:
        """

        # 对话内容
        # 处理历史对话
        formatted_history = self.history_to_text()
        history_input_ids = self.encode(formatted_history, add_bos_token=self.state['add_bos_token'],
                                        truncation_length=self.get_max_prompt_length())

        # 将历史对话向量与当前对话向量拼接
        inputs_embeds = torch.cat((history_input_ids, input_ids), dim=1)

        filler_input_ids = torch.zeros((1, inputs_embeds.shape[1]), dtype=input_ids.dtype).to(self.checkPoint.model.device)
        return inputs_embeds, filler_input_ids

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None) -> str:
        input_ids = self.encode(prompt, add_bos_token=self.state['add_bos_token'],
                                truncation_length=self.get_max_prompt_length())

        # self.history[-self.history_len:] if self.history_len > 0 else []
        output = input_ids[0]
        inputs_embeds, filler_input_ids = self.generate_softprompt_history_tensors(input_ids)
        # self.generate_params.update({'inputs_embeds': inputs_embeds})
        self.generate_params.update({'inputs': inputs_embeds})

        shared.stop_everything = False
        stopped = False
        response_template = _streaming_response_template()
        with self.generate_with_streaming(**self.generate_params) as generator:
            last_reply_index = 0
            # Create a FixedLengthQueue with the desired stop sequence and a maximum length.
            if stop:
                queue = FixedLengthQueue(stop)

            for output in generator:
                new_tokens = len(output) - len(input_ids[0])
                reply = self.decode(output[-new_tokens:])

                new_reply = len(reply) - last_reply_index
                output_reply = reply[-new_reply:]

                if last_reply_index > 0 or new_tokens == self.generate_params['max_new_tokens'] - 1 or stopped:
                    if stop:
                        queue.add(output_reply)
                        pos = queue.contains_stop_sequence()
                        if pos != -1:
                            shared.stop_everything = True
                            stopped = True

                _update_response(response_template, output_reply)
                last_reply_index = len(reply)
                if stopped:
                    break

        response = response_template['text']

        self.history = self.history + [[None, response]]
        return response

    def _generate_answer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False,
                         generate_with_callback: AnswerResultStream = None) -> None:
        self.history = history
        # Create the StoppingCriteriaList with the stopping strings
        stopping_criteria_list = transformers.StoppingCriteriaList()
        # 定义模型stopping_criteria 队列，在每次响应时将 torch.LongTensor, torch.FloatTensor同步到AnswerResult
        listenerQueue = AnswerResultQueueSentinelTokenListenerQueue()
        stopping_criteria_list.append(listenerQueue)
        self.generate_params['stopping_criteria'] = stopping_criteria_list
        # TODO 需要实现chat对话模块和注意力模型，目前_call为langchain的LLM拓展的api，默认为无提示词模式，如果需要操作注意力模型，可以参考chat_glm的实现
        response = self._call(prompt=prompt, stop=['\n###'])
        answer_result = AnswerResult()
        answer_result.history = self.history
        answer_result.listenerToken = listenerQueue.listenerQueue.pop()
        answer_result.llm_output = {"answer": response}
        generate_with_callback(answer_result)
