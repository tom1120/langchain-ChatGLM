from models.loader.args import parser
from models.loader import LoaderLLM

"""打字机效果停止状态"""
stop_everything = False
args = parser.parse_args()

loaderLLM: LoaderLLM = None
