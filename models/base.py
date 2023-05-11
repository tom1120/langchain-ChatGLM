from abc import ABC, abstractmethod
from typing import Optional, List
import traceback
from collections import deque
from queue import Queue
from threading import Thread

import torch
import transformers
from models.loader import LoaderCheckPoint


class ListenerToken:
    """
    观测结果
    """

    input_ids: torch.LongTensor
    _scores: torch.FloatTensor

    def __init__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor):
        self.input_ids = input_ids
        self._scores = _scores


class AnswerResult:
    """
    消息实体
    """
    history: List[List[str]] = []
    llm_output: Optional[dict] = None
    listenerToken: ListenerToken = None


class AnswerResultStream:
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, answerResult: AnswerResult):
        if self.callback_func is not None:
            self.callback_func(answerResult)


class AnswerResultQueueSentinelTokenListenerQueue(transformers.StoppingCriteria):
    """
     定义模型stopping_criteria 监听者，在每次响应时将队列数据同步到AnswerResult
     实现此监听器的目的是，不同模型的预测输出可能不是矢量信息，hf框架可以自定义transformers.StoppingCriteria入参来接收每次预测的Tensor和损失函数，
     输出值可用于 generatorAnswer generate_with_streaming的自定义参数观测，以实现更加精细的控制
    """

    listenerQueue: deque = deque(maxlen=1)

    def __init__(self):
        transformers.StoppingCriteria.__init__(self)

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor, **kwargs) -> bool:
        """
        每次响应时将数据添加到响应队列
        :param input_ids:
        :param _scores:
        :param kwargs:
        :return:
        """
        self.listenerQueue.append(ListenerToken(input_ids=input_ids, _scores=_scores))
        return False


class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}):
        self.mfunc = func
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
            """
            模型输出预测结果收集
            通过 给 StoppingCriteriaList指定模型生成答案时停止的条件。每个 StoppingCriteria 对象表示一个停止条件
            每个StoppingCriteria都会收到相同的预测结果，最终由下层实现类确认是否结束
            因为当前类是迭代器，所以在for in 中执行了break后 __exit__ 方法会被调用，最终stop_now属性会被更新，然后抛出异常结束预测行为
            :param val:
            :return:
            """
            if self.stop_now:
                raise ValueError
            self.q.put(val)

        def gen():
            try:
                ret = self.mfunc(callback=_callback, **self.kwargs)
            except ValueError:
                pass
            except:
                traceback.print_exc()
                pass

            self.q.put(self.sentinel)

        self.thread = Thread(target=gen)
        self.thread.start()

    def __iter__(self):
        return self

    def __next__(self):
        obj = self.q.get(True, None)
        if obj is self.sentinel:
            raise StopIteration
        else:
            return obj

    def __del__(self):
        """
        暂无实现
        :return:
        """
        pass

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """ break 后会执行 """
        self.stop_now = True


class BaseAnswer(ABC):
    """上层业务包装器.用于结果生成统一api调用"""

    @property
    @abstractmethod
    def _check_point(self) -> LoaderCheckPoint:
        """Return _check_point of llm."""

    def generatorAnswer(self, prompt: str,
                        history: List[List[str]] = [],
                        streaming: bool = False):
        def generate_with_callback(callback=None, **kwargs):
            kwargs['generate_with_callback'] = AnswerResultStream(callback_func=callback)
            self._generate_answer(**kwargs)

        def generate_with_streaming(**kwargs):
            return Iteratorize(generate_with_callback, kwargs)

        """
        eos_token_id是指定token（例如，"</s>"），
        用于表示序列的结束。在生成文本任务中，生成器在生成序列时，将不断地生成token，直到生成此特殊的eos_token_id，表示序列生成已经完成。
        在Hugging Face Transformer模型中，eos_token_id是由tokenizer自动添加到输入中的。
        在模型生成输出时，如果模型生成了eos_token_id，则生成过程将停止并返回生成的序列。
        """
        eos_token_ids = [
            self._check_point.tokenizer.eos_token_id] if self._check_point.tokenizer.eos_token_id is not None else []

        with generate_with_streaming(prompt=prompt, history=history, streaming=streaming) as generator:
            for answerResult in generator:
                if answerResult.listenerToken:
                    output = answerResult.listenerToken.input_ids
                yield answerResult

    @abstractmethod
    def _generate_answer(self, prompt: str,
                         history: List[List[str]] = [],
                         streaming: bool = False,
                         generate_with_callback: AnswerResultStream = None) -> None:
        pass
