from pydantic import BaseModel


class ChatGLMModel(BaseModel):
    type: str
    question: str