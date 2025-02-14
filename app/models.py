# app/models.py
from datetime import datetime
from pydantic import BaseModel
from typing import List, Optional, Dict

class UserContext(BaseModel):
    user_id: str
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    preferences: Optional[Dict] = None
    last_interaction: Optional[datetime] = None
    conversation_history: List[Dict] = []
    extracted_details: Dict = {}  # Store any extracted information about the user

class Message(BaseModel):
    role: str
    content: str
    timestamp: Optional[datetime] = None
    user_context: Optional[UserContext] = None

class ChatRequest(BaseModel):
    messages: List[Message]
    max_length: Optional[int] = None
    temperature: Optional[float] = None
    user_id: Optional[str] = None  # Added to track user across sessions

class ChatResponse(BaseModel):
    response: str
    metadata: Optional[Dict] = None
    updated_context: Optional[UserContext] = None  # Return updated context if available

class Document(BaseModel):
    text: str
    metadata: Optional[Dict] = None
    