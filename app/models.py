# app/models.py
from pydantic import BaseModel
from typing import List, Optional, Dict

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    messages: List[Message]
    max_length: Optional[int] = None
    temperature: Optional[float] = None

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict] = None

class Document(BaseModel):
    text: str
    metadata: Optional[Dict] = None
    