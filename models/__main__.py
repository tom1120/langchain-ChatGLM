import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)) + '/../')
import asyncio
from argparse import Namespace
from models.loader.args import parser
from models.loader import LoaderLLM
from models.llama_llm import LLamaLLM
import models.shared as shared

async def dispatch(args: Namespace):
    args_dict = vars(args)

    shared.loaderLLM = LoaderLLM(args_dict)
    llamaLLM = LLamaLLM(shared.loaderLLM)
    llamaLLM._call(prompt="你好")





if __name__ == '__main__':
    args = None
    args = parser.parse_args()

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(dispatch(args))