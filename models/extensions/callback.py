import gc
import traceback
from queue import Queue
from threading import Thread
import threading

from collections import deque
import torch
import transformers

from models.extensions.thread_with_exception import ThreadWithException
import models.shared as shared


class FixedLengthQueue:
    def __init__(self, stop_sequence):
        if stop_sequence is None:
            self.stop_sequence = []
            self.max_length = 0
        elif isinstance(stop_sequence, str):
            self.stop_sequence = [stop_sequence]
            self.max_length = 1
        else:
            self.stop_sequence = stop_sequence
            self.max_length = len(''.join(stop_sequence))

        self.queue = deque(maxlen=self.max_length)

    def add(self, item):
        for char in item:
            self.queue.append(char)

    def contains_stop_sequence(self):
        joined_queue = ''.join(self.queue)
        # Initialize a variable to store the index of the last found stop string
        last_stop_str_index = -1

        # Iterate through the stop string list
        for stop_word in self.stop_sequence:
            # Find the last occurrence of the stop string in the output
            stop_word_index = joined_queue.rfind(stop_word)

            # If the stop string is found, compare the index with the previously found index
            if stop_word_index != -1 and stop_word_index > last_stop_str_index:
                last_stop_str_index = stop_word_index

        # Handle the last found stop string index here
        return last_stop_str_index

    def __repr__(self):
        return str(self.queue)


# Copied from https://github.com/PygmalionAI/gradio-ui/
class _SentinelTokenStoppingCriteria(transformers.StoppingCriteria):

    def __init__(self, sentinel_token_ids: list, starting_idx: int):
        transformers.StoppingCriteria.__init__(self)
        self.sentinel_token_ids = sentinel_token_ids
        self.starting_idx = starting_idx

    def __call__(self, input_ids: torch.LongTensor, _scores: torch.FloatTensor) -> bool:
        for sample in input_ids:
            trimmed_sample = sample[self.starting_idx:]

            for i in range(len(self.sentinel_token_ids)):
                # Can't unfold, output is still too tiny. Skip.
                if trimmed_sample.shape[-1] < self.sentinel_token_ids[i].shape[-1]:
                    continue
                for window in trimmed_sample.unfold(0, self.sentinel_token_ids[i].shape[-1], 1):
                    if torch.all(torch.eq(self.sentinel_token_ids[i][0], window)):
                        return True
        return False


class Stream(transformers.StoppingCriteria):
    def __init__(self, callback_func=None):
        self.callback_func = callback_func

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.callback_func is not None:
            self.callback_func(input_ids[0])
        return False


class Iteratorize:
    """
    Transforms a function that takes a callback
    into a lazy iterator (generator).
    """

    def __init__(self, func, kwargs={}, callback=None):
        self.mfunc = func
        self.c_callback = callback
        self.q = Queue()
        self.sentinel = object()
        self.kwargs = kwargs
        self.stop_now = False

        def _callback(val):
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
        shared.loaderCheckPoint.clear_torch_cache()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop_now = True
        shared.loaderCheckPoint.clear_torch_cache()
