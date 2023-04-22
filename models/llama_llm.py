from langchain.llms.base import LLM

import torch
import transformers
from typing import Optional, List
from models.loader import LoaderLLM
from models.extensions.callback import (Iteratorize, Stream)


class LLamaLLM(LLM):
    llm: LoaderLLM = None
    generate_params: object = {'max_new_tokens': 200,
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
    state: object = {'max_new_tokens': 200,
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

    def __init__(self, llm: LoaderLLM = None):
        super().__init__()
        self.llm = llm

    @property
    def _llm_type(self) -> str:
        return "LLamaLLM"

    def encode(self, prompt, add_special_tokens=True, add_bos_token=True, truncation_length=None):
        input_ids = self.llm.tokenizer.encode(str(prompt), return_tensors='pt',
                                              add_special_tokens=add_special_tokens)
        # This is a hack for making replies more creative.
        if not add_bos_token and input_ids[0][0] == self.llm.tokenizer.bos_token_id:
            input_ids = input_ids[:, 1:]

        # Llama adds this extra token when the first character is '\n', and this
        # compromises the stopping criteria, so we just remove it
        if type(self.llm.tokenizer) is transformers.LlamaTokenizer and input_ids[0][0] == 29871:
            input_ids = input_ids[:, 1:]

        # Handling truncation
        if truncation_length is not None:
            input_ids = input_ids[:, -truncation_length:]

        return input_ids.cuda()

    def decode(self, output_ids):
        reply = self.llm.tokenizer.decode(output_ids, skip_special_tokens=True)
        reply = reply.replace(r'<|endoftext|>', '')
        return reply

    def get_max_prompt_length(self):
        max_length = self.state['truncation_length'] - self.state['max_new_tokens']
        return max_length

    def generate_with_callback(self, callback=None, **kwargs):
        kwargs['stopping_criteria'].append(Stream(callback_func=callback))
        self.llm.clear_torch_cache()
        with torch.no_grad():
            self.llm.model.generate(**kwargs)

    def generate_with_streaming(self, callback=None, **kwargs):
        return Iteratorize(self.generate_with_callback, kwargs, callback)

    def callmessage(self, prompt: str, ):

        input_ids = self.encode(prompt, add_bos_token=self.state['add_bos_token'],
                                truncation_length=self.get_max_prompt_length())
        self.generate_params.update({'inputs': input_ids})

        with self.generate_with_streaming(**self.generate_params) as generator:
            for output in generator:
                new_tokens = len(output) - len(input_ids[0])
                reply = self.decode(output[-new_tokens:])
                print(reply)

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None) -> str:
        input_ids = self.encode(prompt, add_bos_token=self.state['add_bos_token'],
                                truncation_length=self.get_max_prompt_length())
        output = input_ids[0]
        self.generate_params.update({'inputs': input_ids})
        with torch.no_grad():
            output = self.llm.model.generate(**self.generate_params)[0]
            if not self.llm.cpu:
                output = output.cuda()
        new_tokens = len(output) - len(input_ids[0])
        response = self.decode(output[-new_tokens:])
        return response
