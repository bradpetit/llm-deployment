# app/admin.py
from fastapi import APIRouter, Depends, HTTPException, File, UploadFile, Form, Query
from fastapi.security import HTTPBasic, HTTPBasicCredentials
from fastapi.responses import HTMLResponse
import secrets
from pathlib import Path
import os
import json
from typing import List, Optional
from io import BytesIO
import PyPDF2
import docx
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class MetadataExtractor:
    @staticmethod
    def extract_pdf_metadata(file_bytes: bytes) -> dict:
        metadata = {}
        try:
            pdf_file = BytesIO(file_bytes)
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            
            # Extract basic PDF info
            metadata['page_count'] = len(pdf_reader.pages)
            metadata['title'] = pdf_reader.metadata.get('/Title', '')
            metadata['author'] = pdf_reader.metadata.get('/Author', '')
            metadata['creation_date'] = pdf_reader.metadata.get('/CreationDate', '')
            
            # Extract text from first page for topic inference
            first_page_text = pdf_reader.pages[0].extract_text()
            metadata['first_page_preview'] = first_page_text[:200] if first_page_text else ''
            
            # Infer document type and category
            metadata['document_type'] = 'pdf'
            metadata['category'] = MetadataExtractor.infer_category(first_page_text)
            
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
        return metadata

    @staticmethod
    def extract_docx_metadata(file_bytes: bytes) -> dict:
        metadata = {}
        try:
            doc = docx.Document(BytesIO(file_bytes))
            
            # Extract basic DOCX info
            metadata['page_count'] = len(doc.paragraphs)
            core_properties = doc.core_properties
            metadata['title'] = core_properties.title or ''
            metadata['author'] = core_properties.author or ''
            metadata['created'] = str(core_properties.created) if core_properties.created else ''
            
            # Extract text from first paragraph for topic inference
            first_text = doc.paragraphs[0].text if doc.paragraphs else ''
            metadata['first_page_preview'] = first_text[:200]
            
            # Infer document type and category
            metadata['document_type'] = 'docx'
            metadata['category'] = MetadataExtractor.infer_category(first_text)
            
        except Exception as e:
            logger.error(f"Error extracting DOCX metadata: {str(e)}")
        return metadata

    @staticmethod
    def extract_txt_metadata(content: str) -> dict:
        metadata = {}
        try:
            # Basic text analysis
            lines = content.split('\n')
            metadata['line_count'] = len(lines)
            metadata['first_page_preview'] = content[:200]
            
            # Infer document type and category
            metadata['document_type'] = 'txt'
            metadata['category'] = MetadataExtractor.infer_category(content)
            
        except Exception as e:
            logger.error(f"Error extracting TXT metadata: {str(e)}")
        return metadata

    @staticmethod
    def infer_category(text: str) -> str:
        """Infer document category based on content analysis"""
        text = text.lower()
        
        # Define category keywords
        categories = {
            'pricing': ['price', 'cost', 'fee', 'payment', 'rates'],
            'policies': ['policy', 'rule', 'guideline', 'requirement', 'terms'],
            'venue_info': ['capacity', 'space', 'facility', 'amenity', 'feature'],
            'catering': ['food', 'beverage', 'menu', 'catering', 'dining'],
            'contracts': ['contract', 'agreement', 'legal', 'terms', 'conditions'],
            'schedules': ['schedule', 'timeline', 'date', 'time', 'availability'],
            'vendor_info': ['vendor', 'supplier', 'service', 'provider', 'partner']
        }
        
        # Count keyword matches for each category
        category_scores = {cat: sum(1 for kw in kws if kw in text) 
                         for cat, kws in categories.items()}
        
        # Return category with highest score, or 'general' if no matches
        max_score = max(category_scores.values())
        if max_score > 0:
            return max(category_scores.items(), key=lambda x: x[1])[0]
        return 'general'

# Router setup
router = APIRouter()
security = HTTPBasic()

# Auth settings
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME", "admin")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD", "changeme")

def get_current_admin(credentials: HTTPBasicCredentials = Depends(security)):
    is_correct_username = secrets.compare_digest(credentials.username, ADMIN_USERNAME)
    is_correct_password = secrets.compare_digest(credentials.password, ADMIN_PASSWORD)
    
    if not (is_correct_username and is_correct_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username

@router.get("/admin", response_class=HTMLResponse)
async def admin_interface(_: str = Depends(get_current_admin)):
    """Serve the admin interface HTML"""
    html_path = Path(__file__).parent / "templates" / "admin.html"
    with open(html_path, "r") as f:
        return f.read()

@router.get("/admin/documents")
async def list_documents(
    _: str = Depends(get_current_admin),
    page: int = Query(1, ge=1),
    search: str = Query(""),
    page_size: int = Query(10, ge=1, le=100)
):
    """List documents with pagination and search"""
    try:
        from app.main import llm_manager
        documents = llm_manager.list_documents(
            search_term=search,
            page=page,
            page_size=page_size
        )
        return documents
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/admin/upload")
async def upload_files(
    file: UploadFile = File(...),
    metadata_json: Optional[str] = Form(None),
    _: str = Depends(get_current_admin)
):
    """Handle file upload with automatic metadata extraction"""
    try:
        from app.main import llm_manager
        
        # Read file content
        content = await file.read()
        
        # Extract base metadata
        base_metadata = {
            'filename': file.filename,
            'content_type': file.content_type,
            'size': len(content),
            'upload_date': datetime.now().isoformat(),
            'source': 'file_upload'
        }
        
        # Extract format-specific metadata
        if file.filename.lower().endswith('.pdf'):
            file_metadata = MetadataExtractor.extract_pdf_metadata(content)
            text = PyPDF2.PdfReader(BytesIO(content)).pages[0].extract_text()
        elif file.filename.lower().endswith('.docx'):
            file_metadata = MetadataExtractor.extract_docx_metadata(content)
            doc = docx.Document(BytesIO(content))
            text = '\n'.join(paragraph.text for paragraph in doc.paragraphs)
        elif file.filename.lower().endswith('.txt'):
            text = content.decode('utf-8')
            file_metadata = MetadataExtractor.extract_txt_metadata(text)
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file type: {file.filename}"
            )

        # Combine metadata sources
        combined_metadata = {
            **base_metadata,
            **file_metadata
        }
        
        # Add user-provided metadata if any
        if metadata_json:
            try:
                user_metadata = json.loads(metadata_json)
                combined_metadata.update(user_metadata)
            except json.JSONDecodeError:
                logger.warning("Invalid metadata JSON provided")

        # Add to knowledge base
        doc_id = llm_manager.add_to_knowledge_base(
            text=text,
            metadata=combined_metadata
        )

        return {
            'message': 'File uploaded successfully',
            'doc_id': doc_id,
            'metadata': combined_metadata
        }
    except Exception as e:
        logger.error(f"Error during file upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/admin/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    _: str = Depends(get_current_admin)
):
    """Delete a document"""
    try:
        from app.main import llm_manager
        success = llm_manager.delete_document(doc_id)
        if success:
            return {"message": "Document deleted successfully"}
        raise HTTPException(status_code=404, detail="Document not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))