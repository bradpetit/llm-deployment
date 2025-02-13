# app/utils.py
import re
import requests
import logging
from typing import List, Dict, Optional, Any
import chromadb
from app.config import settings
import torch
from sentence_transformers import SentenceTransformer
import numpy as np
import spacy
from chromadb.api.types import Documents, Embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EnhancedEmbedder:
    def __init__(self):
        self.model = SentenceTransformer('all-mpnet-base-v2')
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
        
        self.chunk_size = 512
        self.chunk_overlap = 50

    def __call__(self, input: List[str]) -> List[List[float]]:
        """
        ChromaDB-compatible embedding function
        Args:
            input: List of strings to embed
        Returns:
            List of embeddings as float lists
        """
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in input]
        
        # Generate embeddings
        embeddings = self.model.encode(processed_texts)
        
        # Calculate importance scores for weighting (optional)
        importance_scores = [self.calculate_chunk_importance(text) for text in processed_texts]
        if importance_scores:
            max_score = max(importance_scores)
            importance_scores = [score/max_score for score in importance_scores]
            
            # Weight embeddings by importance (optional)
            weighted_embeddings = []
            for emb, score in zip(embeddings, importance_scores):
                weighted_emb = emb * score
                weighted_embeddings.append(weighted_emb)
            embeddings = weighted_embeddings

        return [embedding.tolist() for embedding in embeddings]

    def preprocess_text(self, text: str) -> str:
        """Clean and normalize text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[""'']', '"', text)
        text = re.sub(r'[–—]', '-', text)
        return text.strip()

    def calculate_chunk_importance(self, text: str) -> float:
        """Calculate importance score for text"""
        doc = self.nlp(text)
        
        entity_score = len(doc.ents) * 0.5
        noun_phrase_score = len(list(doc.noun_chunks)) * 0.3
        keyword_score = sum(1 for token in doc 
                          if token.pos_ in ['NOUN', 'VERB'] 
                          and not token.is_stop) * 0.2
        
        return entity_score + noun_phrase_score + keyword_score
    
class LLMManager:
    def __init__(self):
        self.base_url = "http://ollama.easystreet.studio:11434/api"
        self.model = "llama3.2:latest"
        self.collection = None
        self.chroma_client = None
        self.embedder = None
        self.embedding_function = EnhancedEmbedder()
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
            # Initialize ChromaDB with enhanced embedder
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            self.collection = self.chroma_client.get_or_create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_function
            )
            
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

    def add_to_knowledge_base(self, text: str, metadata: Optional[Dict] = None) -> str:
        """Enhanced document addition with better chunking and metadata"""
        try:
            # Create semantic chunks
            chunks = self.embedding_function.create_semantic_chunks(text)
            
            # Initialize metadata if None
            if metadata is None:
                metadata = {}
            
            doc_ids = []
            for i, chunk in enumerate(chunks):
                # Create chunk-specific metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'original_text_length': len(text),
                    'chunk_length': len(chunk)
                })
                
                # Generate doc ID
                doc_id = f"doc_{len(self.collection.get()['ids']) + 1}_{i}"
                
                # Add to collection
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
    
    def query_knowledge_base(self, query: str, n_results: int = 5) -> List[str]:
        """Enhanced query processing with better ranking"""
        try:
            # Preprocess query
            processed_query = self.embedding_function.preprocess_text(query)
            
            # Get results with metadata
            results = self.collection.query(
                query_texts=[processed_query],
                n_results=n_results * 2,  # Get more results for reranking
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results['documents'][0]:
                return []

            # Combine results with distances for reranking
            combined_results = []
            for doc, meta, dist in zip(
                results['documents'][0],
                results['metadatas'][0],
                results['distances'][0]
            ):
                # Calculate relevance score
                relevance = 1 / (1 + dist)  # Convert distance to similarity
                
                # Get chunk importance (if available)
                importance = meta.get('importance_score', 0.5)
                
                # Calculate final score
                final_score = (relevance * 0.7) + (importance * 0.3)
                
                combined_results.append({
                    'text': doc,
                    'score': final_score,
                    'metadata': meta
                })
            
            # Sort by final score and get top n_results
            combined_results.sort(key=lambda x: x['score'], reverse=True)
            top_results = combined_results[:n_results]
            
            # Extract and return texts
            return [result['text'] for result in top_results]

        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            return []

    def generate_response(self, prompt: str, max_length: Optional[int] = None, temperature: Optional[float] = None) -> str:
        """Generate response with enhanced context integration"""
        try:
            # Get relevant context with improved retrieval
            context = self.query_knowledge_base(prompt)
            
            # Process and combine context
            if context:
                # Remove duplicate information
                seen_content = set()
                unique_context = []
                for c in context:
                    if c not in seen_content:
                        unique_context.append(c)
                        seen_content.add(c)
                
                context_str = "\n\nRelevant Information:\n" + "\n".join(
                    f"- {c}" for c in unique_context
                )
            else:
                context_str = "No specific venue details found for this query."

            # Format prompt with context
            formatted_prompt = self.system_prompt.format(
                context=context_str,
                question=prompt
            )

            # Generate response
            data = {
                "model": self.model,
                "prompt": formatted_prompt,
                "stream": False,
                "temperature": temperature or settings.DEFAULT_TEMPERATURE,
                "max_length": max_length or settings.MAX_LENGTH
            }
            
            response = requests.post(f"{self.base_url}/generate", json=data)
            
            if response.status_code == 200:
                response_text = response.json()["response"].strip()
                return self._clean_response(response_text)
            else:
                raise Exception(f"Error from Ollama API: {response.text}")
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise

    def _clean_response(self, text: str) -> str:
        """Clean up the response text"""
        # Remove conversation markers
        text = re.sub(r'^(Assistant:|User:)\s*', '', text)
        # Clean up any markdown artifacts
        text = re.sub(r'\*\*(Assistant|User):\*\*\s*', '', text)
        return text.strip()
    
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

    def _is_document_relevant(self, document: str, metadata: dict) -> bool:
        """
        Additional filtering logic for documents
        
        :param document: The document text
        :param metadata: The document's metadata
        :return: Boolean indicating if the document is relevant
        """
        # Example filtering criteria
        # You can customize this based on your specific requirements
        
        # Ignore very short documents
        if len(document) < 50:
            return False
        
        # Optional metadata filtering
        if metadata:
            # Example: filter out documents from certain sources
            if metadata.get('source') in ['system_generated', 'test_data']:
                return False
            
            # Example: only include documents from specific categories
            if metadata.get('category'):
                allowed_categories = ['user_guide', 'faq', 'reference']
                if metadata['category'] not in allowed_categories:
                    return False
        
        return True
    
    def _expand_query_terms(self, query: str) -> str:
        """
        Optionally expand query with synonyms or related terms
        
        :param query: Preprocessed query
        :return: Expanded query
        """
        # This is a placeholder. In a real implementation, you might:
        # 1. Use a thesaurus API
        # 2. Implement a custom synonym dictionary
        # 3. Use word embeddings to find related terms
        
        # Simple example with a small synonym dictionary
        synonyms = {
            'price': ['cost', 'pricing', 'fee'],
            'book': ['reserve', 'schedule', 'arrange'],
            # Add more synonym mappings
        }
        
        # Split query into words and expand
        expanded_terms = []
        for term in query.split():
            expanded_terms.append(term)
            expanded_terms.extend(synonyms.get(term, []))
        
        return ' '.join(expanded_terms)