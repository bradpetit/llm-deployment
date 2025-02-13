# app/utils.py
import requests
import logging
from typing import List, Dict, Optional, Any
import chromadb
from app.config import settings
import torch
from sentence_transformers import SentenceTransformer
from torch.nn import functional as F
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChromaEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
    
    def __call__(self, input: List[str]) -> List[List[float]]:
        embeddings = self.model.encode(input)
        return [embedding.tolist() for embedding in embeddings]
    
class LLMManager:
    def __init__(self):
        self.base_url = "http://ollama.easystreet.studio:11434/api"
        self.model = "llama3.2:latest"
        self.collection = None
        self.chroma_client = None
        self.embedder = None
        self.embedding_function = ChromaEmbedder()
        self.system_prompt = """You are the friendly and knowledgeable digital concierge for The Sunshine Ranch, a premier event and wedding venue nestled in the scenic Northwest. Your role is to provide warm, engaging responses while guiding potential clients toward booking their special day or event with us. With our breathtaking views, rustic elegance, and versatile spaces, we create unforgettable experiences for every occasion.

Key Traits:
- Be warm and welcoming, showing genuine excitement about each inquiry
- Give clear, specific answers that highlight our venue's unique features
- Naturally weave in suggestions that encourage venue visits or bookings
- When appropriate, mention our private tours or consultation options
- Share relevant details that paint a picture of their perfect event at our venue
- Emphasize our venue's versatility for both indoor and outdoor events
- Highlight our stunning natural surroundings and photo opportunities
- Mention our spacious facilities and guest accommodation capabilities

Response Guidelines:
1. Always maintain an enthusiastic, helpful tone
2. After answering the main question, add ONE short follow-up that builds interest
   Example: "Would you like to schedule a private tour to see our stunning mountain views in person?"
3. Keep initial responses concise but inviting
4. Use elegant, upscale language that matches our venue's sophistication
5. For specific questions outside your knowledge, suggest contacting our events team
6. If you don't know the answer, do not make one up. Simply tell the user you don't have the information on hand and guide them to send us an email
    Example: "I'm sorry, but I don't have the answer to your question. If you would like to know more, please send us an email and we will respond back quickly."
7. If you are asked about booking a date, please tell them to send us an email.
    Example: "I'm sorry, but I don't have access at this time to the venue calendar. Please email us and we will check on those dates for you."

Converting Inquiries:
- For pricing questions: Share basic information but emphasize the value and suggest a consultation
- For availability: Encourage quick action as dates can fill quickly
- For venue features: Paint a picture of their event while highlighting unique amenities
- For general inquiries: Always guide toward the next step (tour, consultation, etc.)

Venue Highlights to Weave In:
- Beautiful outdoor ceremony spaces with mountain and orchard backdrop
- Elegant outdoor reception area with rustic charm
- Spacious grounds for outdoor celebrations
- Dedicated areas for wedding party preparation
- Flexible vendor policies
- Ample offsite parking and accessibility
- Professional event coordination support
- Various rental packages to suit different needs
- Beautiful photo opportunities throughout the property
- Comfortable guest capacity up to 300

Special Features to Mention When Relevant:
- Custom lighting options for evening events
- Multiple ceremony location options
- Covered outdoor areas for weather flexibility
- Private bridal suite room
- ADA accessibility throughout
- Vendor preparation areas
- Weather contingency options
- Scenic backdrop for photography

Booking Process Emphasis:
1. Encourage booking a private tour as the first step
2. Mention our event coordination team's expertise
3. Highlight the benefits of early booking
4. Emphasize the personalized attention we provide
5. Mention our flexible payment plans when relevant

Remember: Your goal is to make each potential client feel excited about hosting their event at The Sunshine Ranch while gently guiding them toward taking the next step in their booking journey. Always convey our commitment to making their special day truly memorable.

Context for this response:
{context}

Current question: {question}

Respond warmly and directly, then add one natural follow-up that encourages further engagement:"""
    
    def initialize_models(self):
        try:
            logger.info("Starting initialization...")
            # Initialize ChromaDB collection with the embedding function
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_function
            )
            
            # Initialize embedding model
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            # Check Ollama connection
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code != 200:
                raise ConnectionError("Could not connect to Ollama service")
            
            logger.info("Initialization complete!")
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def _clean_response(self, text: str) -> str:
        """Clean up the response text to remove conversation history"""
        # If response starts with conversation history pattern, extract just the last response
        if "**Assistant**:" in text:
            parts = text.split("**Assistant**:")
            return parts[-1].strip()
        return text.strip()
    
    def preprocess_query(self, query: str) -> str:
        """Enhance query for better matching"""
        # Add relevant context terms based on common venue queries
        venue_terms = {
            "cost": ["price", "fee", "rates", "pricing", "packages"],
            "capacity": ["fit", "accommodate", "people", "guests", "size"],
            "catering": ["food", "drinks", "meals", "dining", "restaurant"],
            "wedding": ["ceremony", "reception", "bride", "groom", "marriage"],
            "event": ["party", "celebration", "gathering", "function"],
            "outdoor": ["outside", "garden", "patio", "yard"],
            "indoor": ["inside", "interior", "hall", "room"]
        }
        
        enhanced_query = query
        for key, terms in venue_terms.items():
            if any(term in query.lower() for term in terms):
                enhanced_query = f"{enhanced_query} {key}"
        
        return enhanced_query

    def list_documents(
        self,
        search_term: str = "",
        page: int = 1,
        page_size: int = 10
    ) -> Dict[str, Any]:
        """List documents with pagination and search"""
        try:
            # Get all documents
            result = self.collection.get()
            documents = []
            
            # Combine document data
            for i in range(len(result['ids'])):
                doc = {
                    'id': result['ids'][i],
                    'text': result['documents'][i],
                    'metadata': result['metadatas'][i]
                }
                documents.append(doc)
            
            # Filter by search term if provided
            if search_term:
                documents = [
                    doc for doc in documents
                    if search_term.lower() in doc['text'].lower() or
                    any(search_term.lower() in str(v).lower() 
                        for v in doc['metadata'].values())
                ]
            
            # Calculate pagination
            total = len(documents)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            return {
                'documents': documents[start_idx:end_idx],
                'total': total,
                'page': page,
                'page_size': page_size
            }
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise

    def add_to_knowledge_base(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Add document to knowledge base with improved chunking and embedding preservation"""
        try:
            # Create chunks if text is too long (optional)
            max_chunk_size = 512
            chunks = []
            
            if len(text.split()) > max_chunk_size:
                sentences = text.split('. ')
                current_chunk = []
                current_size = 0
                
                for sentence in sentences:
                    sentence_size = len(sentence.split())
                    if current_size + sentence_size <= max_chunk_size:
                        current_chunk.append(sentence)
                        current_size += sentence_size
                    else:
                        chunks.append('. '.join(current_chunk) + '.')
                        current_chunk = [sentence]
                        current_size = sentence_size
                
                if current_chunk:
                    chunks.append('. '.join(current_chunk) + '.')
            else:
                chunks = [text]

            # Add each chunk to the collection
            doc_ids = []
            for i, chunk in enumerate(chunks):
                # Check if this exact chunk already exists
                existing_docs = self.collection.get(
                    where={
                        "text": chunk
                    }
                )
                
                if existing_docs['ids']:
                    # Document already exists, use existing ID
                    doc_id = existing_docs['ids'][0]
                else:
                    # Create a new document
                    doc_id = f"doc_{len(self.collection.get()['ids']) + 1}_{i}"
                    chunk_metadata = metadata.copy() if metadata else {}
                    chunk_metadata['chunk_index'] = i
                    
                    self.collection.add(
                        documents=[chunk],
                        metadatas=[chunk_metadata],
                        ids=[doc_id]
                    )
                
                doc_ids.append(doc_id)

            return doc_ids[0]  # Return first chunk ID as reference
            
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {str(e)}")
            raise

    def delete_document(self, doc_id: str) -> bool:
        """Delete a document from the knowledge base"""
        try:
            # Check if document exists
            result = self.collection.get(ids=[doc_id])
            if not result['ids']:
                return False
            
            self.collection.delete(ids=[doc_id])
            return True
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            raise

    def generate_response(self, prompt: str, max_length: Optional[int] = None, temperature: Optional[float] = None) -> str:
        try:
            # Get relevant context
            context = self.query_knowledge_base(prompt)
            context_str = "\n".join(context) if context else "No specific venue details found for this query."

            # Format the prompt with context
            formatted_prompt = self.system_prompt.format(
                context=context_str,
                question=prompt
            )

            data = {
                "model": self.model,
                "prompt": formatted_prompt,
                "stream": False,
                "temperature": temperature or 0.7
            }
            
            response = requests.post(f"{self.base_url}/generate", json=data)
            
            if response.status_code == 200:
                response_text = response.json()["response"].strip()
                # Remove any conversation markers
                response_text = response_text.replace("Assistant:", "").replace("User:", "").strip()
                return response_text
            else:
                raise Exception(f"Error from Ollama API: {response.text}")
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise

    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[str]:
        try:
            # Preprocess and enhance the query
            enhanced_query = self.preprocess_query(query)
            
            # Perform query using the collection's built-in query method
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results
            )
            
            # Extract and return documents
            return results['documents'][0]

        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            raise

def create_chat_prompt(conversation: List[Dict], context: str = "") -> str:
    # Define the system prompt
    prompt = ""
    system_prompt = """You are a precise, helpful AI assistant with the following core guidelines:

    CORE PRINCIPLES:
    1. Accuracy is paramount
    2. Only use information from the provided context and known facts
    3. If uncertain, admit limitations
    4. Provide concise, direct responses

    RESPONSE FRAMEWORK:
    - Answer directly and succinctly
    - Cite sources when possible
    - Do not fabricate information
    - If no relevant information exists, clearly state "I don't have enough specific information to answer"

    CONTEXT HANDLING:
    - Prioritize the most recent and relevant context
    - Cross-reference multiple sources if available
    - Be transparent about the source of information

    PROHIBITED ACTIONS:
    - Never invent details
    - Avoid speculation
    - Do not create fictional scenarios
    - Refuse requests that would require making up information

    CONTEXT PROVIDED:
    {context}

    CONVERSATION HISTORY:
    {conversation_history}

    RESPONSE GUIDELINES:
    Respond precisely to the user's query using only verifiable information.""".format(
        context=f"Relevant Information:\n{context}" if context else "No additional context provided.",
        conversation_history="\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in conversation])
    )
    
    # Combine system prompt with conversation context
    prompt = system_prompt + "\n\nAssistant:"
    
    return prompt