import sys

from models.loader.args import parser
from models.loader import LoaderCheckPoint
from configs.model_config import (llm_model_dict, LLM_MODEL)

"""迭代器是否停止状态"""
stop_everything = False
args = parser.parse_args()

loaderCheckPoint: LoaderCheckPoint = None


def loaderLLM(no_remote_model, use_ptuning_v2):
    """
    初始化LLM
    :param no_remote_model:  remote in the model on loader checkpoint, if your load local model to add the ` --no-remote-model
    :param use_ptuning_v2: Use p-tuning-v2 PrefixEncoder
    :return:
    """
    llm_model_info = llm_model_dict[LLM_MODEL]
    loaderCheckPoint.model_name = llm_model_info['path']
    loaderCheckPoint.no_remote_model = no_remote_model
    loaderCheckPoint.use_ptuning_v2 = use_ptuning_v2
    loaderCheckPoint.reload_model()
    provides_class = getattr(sys.modules['models'], llm_model_info['provides'])
    modelInsLLM = provides_class(llm=loaderCheckPoint)

    return modelInsLLM
