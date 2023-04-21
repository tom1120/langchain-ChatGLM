import gc
import json
import os
import re
import time
import zipfile
from pathlib import Path

import numpy as np
import torch
import transformers
from accelerate import infer_auto_device_map, init_empty_weights
from transformers import (AutoConfig, AutoModel, AutoModelForCausalLM,
                          AutoTokenizer, BitsAndBytesConfig, LlamaTokenizer)


class LoaderLLM:
    """
    加载自定义 model
    """
    model_name: str = None
    tokenizer: object = None
    model: object = None
    model_dir: str = None
    cpu: bool = False
    gpu_memory: object = None
    cpu_memory: object = None
    auto_devices: object = True
    load_in_8bit: bool = False
    trust_remote_code: bool = False
    is_llamacpp: bool = False
    bf16: bool = False
    params: object = None

    def __init__(self, params: dict = None):
        """
        模型初始化
        :param params:
        """
        self.params = params or {}
        self.model_name = params.get('model', '')
        self.model = None
        self.tokenizer = None
        self.model_dir = params.get('model_dir', '')
        self.cpu = params.get('cpu', False)
        self.gpu_memory = params.get('gpu_memory', None)
        self.cpu_memory = params.get('cpu_memory', None)
        self.auto_devices = params.get('auto_devices', True)
        self.load_in_8bit = params.get('load_in_8bit', False)
        self.trust_remote_code = params.get('trust_remote_code', False)
        self.bf16 = params.get('bf16', False)
        self.reload_model()

    def _load_model(self, model_name):
        """
        加载自定义位置的model
        :param model_name:
        :return:
        """
        print(f"Loading {model_name}...")
        t0 = time.time()

        self.is_llamacpp = len(list(Path(f'{self.model_dir}/{model_name}').glob('ggml*.bin'))) > 0
        if 'chatglm' in model_name.lower():
            LoaderClass = AutoModel
            trust_remote_code = self.trust_remote_code
        else:
            LoaderClass = AutoModelForCausalLM
            trust_remote_code = False

        # Load the model in simple 16-bit mode by default
        if not any([self.cpu, self.load_in_8bit, self.auto_devices, self.gpu_memory is not None, self.cpu_memory is not None, self.is_llamacpp]):
            model = LoaderClass.from_pretrained(Path(f"{self.model_dir}/{model_name}"), low_cpu_mem_usage=True, torch_dtype=torch.bfloat16 if self.bf16 else torch.float16, trust_remote_code=trust_remote_code)
            if torch.has_mps:
                device = torch.device('mps')
                model = model.to(device)
            else:
                model = model.cuda()

        elif self.is_llamacpp:
            from models.extensions.llamacpp_model_alternative import LlamaCppModel

            model_file = list(Path(f'{self.model_dir}/{model_name}').glob('ggml*.bin'))[0]
            print(f"llama.cpp weights detected: {model_file}\n")

            model, tokenizer = LlamaCppModel.from_pretrained(model_file)
            return model, tokenizer

        # Custom
        else:
            params = {"low_cpu_mem_usage": True}
            if not any((self.cpu, torch.cuda.is_available(), torch.has_mps)):
                print("Warning: torch.cuda.is_available() returned False.\nThis means that no GPU has been detected.\nFalling back to CPU mode.\n")
                self.cpu = True

            if self.cpu:
                params["torch_dtype"] = torch.float32
            else:
                params["device_map"] = 'auto'
                params["trust_remote_code"] = trust_remote_code
                if self.load_in_8bit and any((self.auto_devices, self.gpu_memory)):
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True, llm_int8_enable_fp32_cpu_offload=True)
                elif self.load_in_8bit:
                    params['quantization_config'] = BitsAndBytesConfig(load_in_8bit=True)
                elif shared.args.bf16:
                    params["torch_dtype"] = torch.bfloat16
                else:
                    params["torch_dtype"] = torch.float16

                if self.gpu_memory:
                    memory_map = list(map(lambda x: x.strip(), self.gpu_memory))
                    max_cpu_memory = self.cpu_memory.strip() if self.cpu_memory is not None else '99GiB'
                    max_memory = {}
                    for i in range(len(memory_map)):
                        max_memory[i] = f'{memory_map[i]}GiB' if not re.match('.*ib$', memory_map[i].lower()) else memory_map[i]
                    max_memory['cpu'] = max_cpu_memory
                    params['max_memory'] = max_memory
                elif self.auto_devices:
                    total_mem = (torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
                    suggestion = round((total_mem - 1000) / 1000) * 1000
                    if total_mem - suggestion < 800:
                        suggestion -= 1000
                    suggestion = int(round(suggestion / 1000))
                    print(f"\033[1;32;1mAuto-assiging --gpu-memory {suggestion} for your GPU to try to prevent out-of-memory errors.\nYou can manually set other values.\033[0;37;0m")

                    max_memory = {0: f'{suggestion}GiB', 'cpu': f'{self.cpu_memory or 99}GiB'}
                    params['max_memory'] = max_memory

            checkpoint = Path(f'{self.model_dir}/{model_name}')

            if self.load_in_8bit and params.get('max_memory', None) is not None and params['device_map'] == 'auto':
                config = AutoConfig.from_pretrained(checkpoint)
                with init_empty_weights():
                    model = AutoModelForCausalLM.from_config(config)
                model.tie_weights()
                params['device_map'] = infer_auto_device_map(
                    model,
                    dtype=torch.int8,
                    max_memory=params['max_memory'],
                    no_split_module_classes=model._no_split_modules
                )

            model = AutoModelForCausalLM.from_pretrained(checkpoint, **params)

        # Loading the tokenizer
        if type(model) is transformers.LlamaForCausalLM:
            tokenizer = LlamaTokenizer.from_pretrained(Path(f"{self.model_dir}/{model_name}/"), clean_up_tokenization_spaces=True)
            # Leaving this here until the LLaMA tokenizer gets figured out.
            # For some people this fixes things, for others it causes an error.
            try:
                tokenizer.eos_token_id = 2
                tokenizer.bos_token_id = 1
                tokenizer.pad_token_id = 0
            except:
                pass
        else:
            tokenizer = AutoTokenizer.from_pretrained(Path(f"{self.model_dir}/{model_name}/"), trust_remote_code=trust_remote_code)

        print(f"Loaded the model in {(time.time()-t0):.2f} seconds.")
        return model, tokenizer

    def clear_torch_cache(self):
        gc.collect()
        if not self.cpu:
            torch.cuda.empty_cache()

    def unload_model(self):
        self.model = self.tokenizer = None
        self.clear_torch_cache()

    def reload_model(self):
        self.unload_model()
        self.model, self.tokenizer = self._load_model(self.model_name)
