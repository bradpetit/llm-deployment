# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.models import ChatRequest, ChatResponse, Document
from app.utils import LLMManager
from app.admin import router as admin_router
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LLM API with RAG")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(admin_router)

llm_manager = LLMManager()

@app.on_event("startup")
async def startup_event():
    llm_manager.initialize_models()

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy"}

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    try:
        # Convert messages to list of dicts for processing
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get last user message for RAG
        last_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            None
        )

        # Query knowledge base if there's a user message
        context = []
        if last_user_message:
            context = llm_manager.query_knowledge_base(last_user_message)
        
        # Convert context to string
        context_str = "\n".join(context) if context else "No specific venue details found for this query."

        # Generate response using the existing system prompt method
        response = llm_manager.generate_response(
            prompt=last_user_message,
            max_length=request.max_length,
            temperature=request.temperature
        )
        
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
    
# New Admin Endpoints
@app.get("/admin/collections")
async def list_collections():
    """List all available collections"""
    return {"collections": llm_manager.list_collections()}

@app.post("/admin/collections/{name}")
async def create_collection(name: str):
    """Create a new collection"""
    success = llm_manager.create_collection(name)
    if success:
        return {"message": f"Collection {name} created successfully"}
    raise HTTPException(status_code=500, detail="Failed to create collection")

@app.delete("/admin/collections/{name}")
async def delete_collection(name: str):
    """Delete a collection"""
    success = llm_manager.delete_collection(name)
    if success:
        return {"message": f"Collection {name} deleted successfully"}
    raise HTTPException(status_code=500, detail="Failed to delete collection")

@app.get("/admin/documents")
async def list_documents(collection: str = "knowledge_base"):
    """List all documents in a collection"""
    return {"documents": llm_manager.list_documents(collection)}

@app.delete("/admin/documents/{doc_id}")
async def delete_document(doc_id: str, collection: str = "knowledge_base"):
    """Delete a specific document"""
    success = llm_manager.delete_document(doc_id, collection)
    if success:
        return {"message": f"Document {doc_id} deleted successfully"}
    raise HTTPException(status_code=500, detail="Failed to delete document")

@app.post("/admin/test-query")
async def test_query(query: str, n_results: int = 3):
    """Test a query and return results with relevance scores"""
    try:
        results = llm_manager.test_query(query, n_results)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))