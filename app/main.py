# app/main.py
import traceback
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
        logger.info(f"Received chat request with {len(request.messages)} messages")
        
        # Convert messages to list of dicts for processing
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Get last user message
        last_user_message = next(
            (msg["content"] for msg in reversed(messages) if msg["role"] == "user"),
            None
        )
        
        if not last_user_message:
            raise HTTPException(status_code=400, detail="No user message found")

        # Generate response with user context
        response, updated_context = llm_manager.generate_response(
            prompt=last_user_message,
            message_history=messages,
            user_id=request.user_id,  # Pass the user_id if provided
            max_length=request.max_length,
            temperature=request.temperature
        )
        
        return ChatResponse(
            response=response,
            updated_context=updated_context
        )
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        error_detail = str(e) if str(e) else "Internal server error occurred"
        raise HTTPException(status_code=500, detail=error_detail)

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
    
@app.get("/status")
async def check_status():
    """Check the status of the system"""
    try:
        collection_status = llm_manager.check_collection_status()
        return {
            "status": "healthy",
            "collection": collection_status
        }
    except Exception as e:
        logger.error(f"Error checking status: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))