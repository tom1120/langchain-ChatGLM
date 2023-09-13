import argparse
import os

os.environ['http_proxy'] = 'http://127.0.0.1:7890'
os.environ['https_proxy'] = 'http://127.0.0.1:7890'

# windows默认地址 C:\Users\Administrator\.cache
# os.environ["cache_dir"] = "F:/huggingface/hub"
os.environ["HF_HOME"] = "F:/huggingface"

import uvicorn
from fastapi import Body, FastAPI, File, Form, Query, UploadFile, WebSocket
from fastapi.middleware.cors import CORSMiddleware

from app.controller.model_controller import stream_chat, document, chat, upload_file, upload_files, local_doc_chat, \
    list_docs, delete_docs, generate_embeddings, tokenize, stream_chat_local, stream_chat_sse
from app.model.model_res import BaseResponse, ChatMessage, ListDocsResponse
# 模型初始化
from configs.model_config import OPEN_CROSS_DOMAIN, LLM_MODEL, EMBEDDING_MODEL, EMBEDDING_DEVICE, LLM_HISTORY_LEN, \
    VECTOR_SEARCH_TOP_K


def main():
    global app
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()

    app = FastAPI()
    # Add CORS middleware to allow all origins
    # 在config.py中设置OPEN_DOMAIN=True，允许跨域
    # set OPEN_DOMAIN=True in config.py to allow cross-domain
    if OPEN_CROSS_DOMAIN:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )
    app.websocket("/local_doc_qa/stream-chat/{knowledge_base_id}")(stream_chat)
    app.websocket("/stream_chat_local")(stream_chat_local)

    app.get("/", response_model=BaseResponse)(document)

    app.post("/chat", response_model=ChatMessage)(chat)

    app.post("/local_doc_qa/upload_file", response_model=BaseResponse)(upload_file)
    app.post("/local_doc_qa/upload_files", response_model=BaseResponse)(upload_files)
    app.post("/local_doc_qa/local_doc_chat", response_model=ChatMessage)(local_doc_chat)
    app.get("/local_doc_qa/list_files", response_model=ListDocsResponse)(list_docs)
    app.delete("/local_doc_qa/delete_file", response_model=BaseResponse)(delete_docs)

    app.post("/local_doc_qa/generate_embeddings")(generate_embeddings)
    app.post("/local_doc_qa/tokenize")(tokenize)

    app.route("/stream_param", methods=['GET', 'POST', 'OPTIONs'])(stream_chat_sse)

    uvicorn.run(app, host=args.host, port=args.port)
    # uvicorn.run(app="__main__:app", host=args.host, port=args.port, reload=True, reload_dirs=["app"])


if __name__ == "__main__":
    main()
