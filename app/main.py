# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models import ChatRequest, ChatResponse, Document
from app.utils import LLMManager, create_chat_prompt
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM API with RAG")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize LLM Manager
llm_manager = LLMManager()

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    llm_manager.initialize_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        logger.info("Received chat request")
        
        # Get last user message for RAG
        last_user_message = next(
            (msg.content for msg in reversed(request.messages) if msg.role == "user"),
            None
        )

        # Query knowledge base if there's a user message
        context = ""
        if last_user_message:
            relevant_docs = llm_manager.query_knowledge_base(last_user_message)
            context = "\n".join(relevant_docs)

        # Create prompt with conversation history and context
        prompt = create_chat_prompt(
            conversation=request.messages,
            context=context
        )

        # Generate response
        response = llm_manager.generate_response(
            prompt=prompt,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        # Extract just the assistant's response
        response = response.split("Assistant:")[-1].strip()
        
        logger.info("Generated response successfully")
        return ChatResponse(response=response)

    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/documents")
async def add_document(document: Document):
    """Add a document to the knowledge base"""
    try:
        doc_id = llm_manager.add_to_knowledge_base(
            text=document.text,
            metadata=document.metadata
        )
        return {"message": "Document added successfully", "doc_id": doc_id}
    except Exception as e:
        logger.error(f"Error adding document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True
    )