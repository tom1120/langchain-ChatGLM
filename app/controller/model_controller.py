import json
import logging
import os
import shutil
import uuid
from typing import Annotated, List, Optional

from fastapi import UploadFile, File, Form, Query, Body, WebSocket,Request
from sse_starlette import EventSourceResponse

from starlette.responses import RedirectResponse

from app.model.chat_model import ChatGLMModel
from app.model.model_res import BaseResponse, ListDocsResponse, ChatMessage
from app.service.model_init_service import local_doc_qa
from app.tool.logging_tool import log
from app.tool.path_tool import get_folder_path, get_vs_path, get_file_path
from configs.model_config import UPLOAD_ROOT_PATH, VS_ROOT_PATH


async def upload_file(
        file: UploadFile = File(description="A single binary file"),
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)

    file_content = await file.read()  # 读取上传文件的内容

    file_path = os.path.join(saved_path, file.filename)
    if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
        file_status = f"文件 {file.filename} 已存在。"
        return BaseResponse(code=200, msg=file_status)

    with open(file_path, "wb") as f:
        f.write(file_content)

    vs_path = get_vs_path(knowledge_base_id)
    vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store([file_path], vs_path)
    if len(loaded_files) > 0:
        file_status = f"文件 {file.filename} 已上传至新的知识库，并已加载知识库，请开始提问。"
        return BaseResponse(code=200, msg=file_status)
    else:
        file_status = "文件上传失败，请重新上传"
        return BaseResponse(code=500, msg=file_status)


async def upload_files(
        files: Annotated[
            List[UploadFile], File(description="Multiple files as UploadFile")
        ],
        knowledge_base_id: str = Form(..., description="Knowledge Base Name", example="kb1"),
):
    saved_path = get_folder_path(knowledge_base_id)
    if not os.path.exists(saved_path):
        os.makedirs(saved_path)
    filelist = []
    for file in files:
        file_content = ''
        file_path = os.path.join(saved_path, file.filename)
        file_content = file.file.read()
        if os.path.exists(file_path) and os.path.getsize(file_path) == len(file_content):
            continue
        with open(file_path, "ab+") as f:
            f.write(file_content)
        filelist.append(file_path)
    if filelist:
        vs_path, loaded_files = local_doc_qa.init_knowledge_vector_store(filelist, get_vs_path(knowledge_base_id))
        if len(loaded_files):
            file_status = f"已上传 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 至知识库，并已加载知识库，请开始提问"
            return BaseResponse(code=200, msg=file_status)
    file_status = "文件未成功加载，请重新上传文件"
    return BaseResponse(code=500, msg=file_status)


async def list_docs(
        knowledge_base_id: Optional[str] = Query(default=None, description="Knowledge Base Name", example="kb1")
):
    if knowledge_base_id:
        local_doc_folder = get_folder_path(knowledge_base_id)
        if not os.path.exists(local_doc_folder):
            return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found", "data": []}
        all_doc_names = [
            doc
            for doc in os.listdir(local_doc_folder)
            if os.path.isfile(os.path.join(local_doc_folder, doc))
        ]
        return ListDocsResponse(data=all_doc_names)
    else:
        if not os.path.exists(UPLOAD_ROOT_PATH):
            all_doc_ids = []
        else:
            all_doc_ids = [
                folder
                for folder in os.listdir(UPLOAD_ROOT_PATH)
                if os.path.isdir(os.path.join(UPLOAD_ROOT_PATH, folder))
            ]

        return ListDocsResponse(data=all_doc_ids)


async def delete_docs(
        knowledge_base_id: str = Form(...,
                                      description="Knowledge Base Name(注意此方法仅删除上传的文件并不会删除知识库(FAISS)内数据)",
                                      example="kb1"),
        doc_name: Optional[str] = Form(
            None, description="doc name", example="doc_name_1.pdf"
        ),
):
    if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, knowledge_base_id)):
        return {"code": 1, "msg": f"Knowledge base {knowledge_base_id} not found"}
    if doc_name:
        doc_path = get_file_path(knowledge_base_id, doc_name)
        if os.path.exists(doc_path):
            os.remove(doc_path)
        else:
            return {"code": 1, "msg": f"document {doc_name} not found"}

        remain_docs = await list_docs(knowledge_base_id)
        if remain_docs.code == 200 and len(remain_docs.data) == 0:
            shutil.rmtree(get_folder_path(knowledge_base_id), ignore_errors=True)
        else:
            local_doc_qa.init_knowledge_vector_store(
                get_folder_path(knowledge_base_id), get_vs_path(knowledge_base_id)
            )
    else:
        shutil.rmtree(get_folder_path(knowledge_base_id))
    return BaseResponse()


async def local_doc_chat(
        knowledge_base_id: str = Body(..., description="Knowledge Base Name", example="kb1"),
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)
    if not os.path.exists(vs_path):
        raise ValueError(f"Knowledge base {knowledge_base_id} not found")

    for resp, history in local_doc_qa.get_knowledge_based_answer(
            query=question, vs_path=vs_path, chat_history=history, streaming=True
    ):
        pass
    source_documents = [
        f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
        f"""相关度：{doc.metadata['score']}\n\n"""
        for inum, doc in enumerate(resp["source_documents"])
    ]

    return ChatMessage(
        question=question,
        response=resp["result"],
        history=history,
        source_documents=source_documents,
    )


async def chat(
        question: str = Body(..., description="Question", example="工伤保险是什么？"),
        history: List[List[str]] = Body(
            [],
            description="History of previous questions and answers",
            example=[
                [
                    "工伤保险是什么？",
                    "工伤保险是指用人单位按照国家规定，为本单位的职工和用人单位的其他人员，缴纳工伤保险费，由保险机构按照国家规定的标准，给予工伤保险待遇的社会保险制度。",
                ]
            ],
        ),
):
    for resp, history in local_doc_qa.llm._call(
            prompt=question, history=history, streaming=True
    ):
        pass

    return ChatMessage(
        question=question,
        response=resp,
        history=history,
        source_documents=[],
    )


STREAM_DELAY = 1  # second
RETRY_TIMEOUT = 180000  # milisecond
async def stream_chat_sse(request: Request):
    message_id = str(uuid.uuid4())

    def start():
        yield {
            "event": "start",
            "id": message_id,
            "retry": RETRY_TIMEOUT,
            "data": "data is start"
        }

    def exception(err: str):
        yield {
            "event": "exception",
            "id": message_id,
            "retry": RETRY_TIMEOUT,
            "data": err
        }

    def new_messages(st: List[str]):
        # Add logic here to check for new messages
        for s in st:
            yield {'message_id': message_id, 'message_content': s}
        yield {'message_id': message_id, 'message_content': '[ENDEND]'}

    async def event_generator(data: dict):

        if await request.is_disconnected():
            yield {
                "event": "exception",
                "id": message_id,
                "retry": RETRY_TIMEOUT,
                "data": "is_disconnected"
            }
            return

        log.logger.info(data)
        match data['type']:
            case 'chatGLM':
                question = data['question']
                history = []
                turn = 1
                yield {
                            "event": "new_message",
                            "id": message_id,
                            "retry": RETRY_TIMEOUT,
                            "data": "{{start123}}"
                        }
                last_print_len = 0
                for resp, history in local_doc_qa.llm._call(prompt=question, history=history, streaming=True):
                    yield {
                        "event": "new_message",
                        "id": message_id,
                        "retry": RETRY_TIMEOUT,
                        "data": resp
                    }


            case _:
                # Checks for new messages and return them to client if any
                for line in new_messages(['s', 't', 'u', 'd', 'y']):
                    if line["message_content"] != '[ENDEND]':
                        yield {
                            "event": "new_message",
                            "id": message_id,
                            "retry": RETRY_TIMEOUT,
                            "data": line["message_content"]
                        }
                    else:
                        yield {
                            "event": "end",
                            "id": message_id,
                            "retry": RETRY_TIMEOUT,
                            "data": line["message_content"]
                        }
                log.logger.info("不支持的对话类型")

    method = request.method
    if method == "POST":
        try:
            d = await request.json()
            log.logger.info(d)
            return EventSourceResponse(event_generator(d))
        except Exception as e:
            log.logger.exception(str(e))
            return EventSourceResponse(exception(str(e)))
    else:
        return EventSourceResponse(start())



# 后续再完善ws支持
async def stream_chat_local(websocket: WebSocket):
    await websocket.accept()
    history = []
    turn = 1
    while True:
        data_text = await websocket.receive_text()
        log.logger.info(data_text)
        if data_text is None or len(data_text) == 0:
            break
        data = json.loads(data_text)
        log.logger.debug(data)
        chat_content = data["content"]
        question = chat_content["question"]
        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp, history in local_doc_qa.llm._call(prompt=question, history=history, streaming=True):
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])
        source_documents = []
        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def stream_chat(websocket: WebSocket, knowledge_base_id: str):
    await websocket.accept()
    vs_path = os.path.join(VS_ROOT_PATH, knowledge_base_id)

    if not os.path.exists(vs_path):
        await websocket.send_json({"error": f"Knowledge base {knowledge_base_id} not found"})
        await websocket.close()
        return

    history = []
    turn = 1
    while True:
        question = await websocket.receive_text()
        await websocket.send_json({"question": question, "turn": turn, "flag": "start"})

        last_print_len = 0
        for resp, history in local_doc_qa.get_knowledge_based_answer(
                query=question, vs_path=vs_path, chat_history=history, streaming=True
        ):
            await websocket.send_text(resp["result"][last_print_len:])
            last_print_len = len(resp["result"])

        source_documents = [
            f"""出处 [{inum + 1}] {os.path.split(doc.metadata['source'])[-1]}：\n\n{doc.page_content}\n\n"""
            f"""相关度：{doc.metadata['score']}\n\n"""
            for inum, doc in enumerate(resp["source_documents"])
        ]

        await websocket.send_text(
            json.dumps(
                {
                    "question": question,
                    "turn": turn,
                    "flag": "end",
                    "sources_documents": source_documents,
                },
                ensure_ascii=False,
            )
        )
        turn += 1


async def document():
    return RedirectResponse(url="/docs")


# 嵌词向量生成
async def generate_embeddings(texts: List[str]):
    return local_doc_qa.embeddings.embed_documents(texts)


async def tokenize(texts: List[str]):
    return local_doc_qa.llm.tokenize(texts)
