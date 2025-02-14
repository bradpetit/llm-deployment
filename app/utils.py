# app/utils.py
import json
import re
import requests
import logging
import numpy as np
import chromadb
import torch
from typing import List, Dict, Optional, Any, Union
from app.config import settings
from datetime import datetime
from sentence_transformers import SentenceTransformer
from .enhanced_embedding import EnhancedEmbedder
from .utils_similarity import SimilarityUtils

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self):
        self.base_url = "http://ollama.easystreet.studio:11434/api"
        self.model = "llama3.2:latest"
        self.collection = None
        self.chroma_client = None
        self.embedder = None
        self.embedding_function = EnhancedEmbedder()
        self.system_prompt = """You are the welcoming digital concierge for The Sunshine Ranch, a premier Northwest wedding and event venue. Use the following retrieved context to answer the question. If you don't know the answer, just say you don't know. Use three sentences maximum and keep the answer concise.

Core Guidelines:
- Highlight relevant venue features and capabilities
- Guide conversations toward bookings, tours, or consultations
- Keep responses concise, elegant, and upscale in tone
- For unavailable information, direct to thesunshineranch.events@gmail.com

Key Features to Reference:
- Outdoor ceremony spaces with mountain/orchard backdrop
- Elegant reception areas with rustic charm
- Weather-flexible indoor/outdoor options
- Private bridal suite and prep areas
- Professional event coordination
- Custom lighting and décor options
- Comprehensive vendor amenities
- Easy accessibility and ample parking

Standard Responses:
1. For date inquiries: "To check specific dates, please email thesunshineranch.events@gmail.com or call 1 (509)387-1279"
2. For detailed pricing: "Our packages are between $4999.00 for up to 50 and $7999.00 for up to 300 people, I'd love to arrange a consultation to discuss your specific needs. Please email us at thesunshineranch.events@gmail.com"
3. For availability: "Our dates are filling quickly. Let's schedule a tour to explore your options. Please email us at thesunshineranch.events@gmail.com or call 1 (509)387-1270"

Remember: Share venue highlights naturally, and sometimes conclude with ONE engagement prompt (e.g., "Would you like to schedule a private tour? Send us an email.")

Context: {context}
Question: {question}"""

    def initialize_models(self):
        # self.reset_and_initialize_chroma()
        
        try:
            logger.info("Starting initialization...")
            # Initialize ChromaDB with metadata filtering capability
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
    
    def reset_and_initialize_chroma(self):
        """Reset ChromaDB and initialize with MPNet embeddings"""
        try:
            logger.info("Starting ChromaDB reset and initialization...")
            
            # 1. First, close any existing ChromaDB connections
            if self.collection is not None:
                # Delete all documents from existing collection
                try:
                    results = self.collection.get()
                    if results['ids']:
                        self.collection.delete(ids=results['ids'])
                    logger.info("Existing documents deleted")
                except Exception as e:
                    logger.warning(f"Error clearing collection: {str(e)}")
            
            if self.chroma_client is not None:
                try:
                    self.chroma_client.reset()
                    logger.info("ChromaDB client reset")
                except Exception as e:
                    logger.warning(f"Error resetting client: {str(e)}")
            
            # 2. Initialize new ChromaDB client
            self.chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            
            # 3. Delete and recreate collection
            try:
                self.chroma_client.delete_collection("knowledge_base")
                logger.info("Old collection deleted")
            except Exception as e:
                logger.info("No existing collection to delete")
            
            # 4. Create new collection with MPNet embedding function
            self.collection = self.chroma_client.create_collection(
                name="knowledge_base",
                embedding_function=self.embedding_function,
                metadata={"description": "Knowledge base using MPNet embeddings (768 dimensions)"}
            )
            
            logger.info("ChromaDB reset and initialized successfully with MPNet embeddings")
            return True
            
        except Exception as e:
            logger.error(f"Error during ChromaDB reset: {str(e)}")
            raise

    # Test function to verify embedding dimensions
    def verify_embedding_dimensions(self):
        """Verify that the embeddings are 768-dimensional (MPNet)"""
        try:
            # Create a test embedding
            test_text = ["This is a test document"]
            test_embedding = self.embedding_function(test_text)[0]
            
            # Check dimensions
            embedding_dim = len(test_embedding)
            logger.info(f"Embedding dimensions: {embedding_dim}")
            
            if embedding_dim != 768:
                raise ValueError(f"Unexpected embedding dimensions: {embedding_dim} (expected 768)")
            
            return True
        except Exception as e:
            logger.error(f"Error verifying embedding dimensions: {str(e)}")
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

    def _prepare_metadata_for_chroma(self, metadata: Dict) -> Dict:
        """
        Convert metadata values to ChromaDB-compatible types
        """
        prepared_metadata = {}
        for key, value in metadata.items():
            if isinstance(value, (str, int, float, bool)):
                prepared_metadata[key] = value
            elif isinstance(value, list):
                # Convert lists to comma-separated strings
                prepared_metadata[key] = ','.join(str(item) for item in value)
            elif isinstance(value, dict):
                # Convert dictionaries to JSON strings
                prepared_metadata[key] = json.dumps(value)
            elif value is None:
                # Handle None values
                prepared_metadata[key] = ''
            else:
                # Convert any other types to strings
                prepared_metadata[key] = str(value)
        return prepared_metadata

    def add_to_knowledge_base(
        self, 
        text: str, 
        metadata: Optional[Dict] = None,
        chunk_size: int = 512,
        chunk_overlap: int = 50
    ) -> Union[str, List[str]]:
        """
        Add document to knowledge base with improved chunking and metadata handling
        """
        try:
            # Validate metadata
            if not metadata:
                metadata = {}
            if 'timestamp' not in metadata:
                metadata['timestamp'] = datetime.utcnow().isoformat()
            if 'status' not in metadata:
                metadata['status'] = 'active'

            # Create chunks if text is too long
            chunks = []
            if len(text.split()) > chunk_size:
                # Improved chunking with overlap
                words = text.split()
                for i in range(0, len(words), chunk_size - chunk_overlap):
                    chunk = ' '.join(words[i:i + chunk_size])
                    chunks.append(chunk)
            else:
                chunks = [text]

            # Add each chunk to the collection
            doc_ids = []
            for i, chunk in enumerate(chunks):
                # Create chunk-specific metadata
                chunk_metadata = metadata.copy()
                chunk_metadata.update({
                    'chunk_index': i,
                    'total_chunks': len(chunks),
                    'chunk_size': len(chunk.split()),
                    'is_chunked': len(chunks) > 1
                })
                
                # Prepare metadata for ChromaDB
                prepared_metadata = self._prepare_metadata_for_chroma(chunk_metadata)
                
                # Generate or use provided document ID
                doc_id = chunk_metadata.get('document_id', 
                    f"doc_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{i}")
                
                # Add to collection
                self.collection.add(
                    documents=[chunk],
                    metadatas=[prepared_metadata],
                    ids=[doc_id]
                )
                doc_ids.append(doc_id)

            return doc_ids[0] if len(doc_ids) == 1 else doc_ids
            
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {str(e)}")
            raise
    
    def _convert_stored_metadata(self, metadata: Dict) -> Dict:
        """
        Convert stored metadata back to original types
        """
        converted = {}
        for key, value in metadata.items():
            if key in ['categories', 'tags', 'keywords']:
                # Convert comma-separated strings back to lists
                converted[key] = value.split(',') if value else []
            elif key in ['properties', 'config', 'settings']:
                # Convert JSON strings back to dictionaries
                try:
                    converted[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    converted[key] = value
            else:
                converted[key] = value
        return converted

    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[str]:
        try:
            # Enhance query first
            enhanced_query = self._enhance_query(query)
            
            # Get initial results
            results = self.collection.query(
                query_texts=[enhanced_query],
                n_results=n_results * 2  # Get more results initially
            )
            
            if not results or not results['documents']:
                return []
            
            # Process and rerank results
            documents = results['documents'][0]
            metadatas = results['metadatas'][0]
            distances = results['distances'][0]
            
            # Rerank based on both semantic similarity and content relevance
            reranked_results = []
            for doc, meta, dist in zip(documents, metadatas, distances):
                relevance_score = self._calculate_relevance_score(
                    query=query,
                    document=doc,
                    distance=dist,
                    metadata=meta
                )
                reranked_results.append((doc, relevance_score))
            
            # Sort by relevance score and take top n_results
            reranked_results.sort(key=lambda x: x[1], reverse=True)
            return [doc for doc, _ in reranked_results[:n_results]]
            
        except Exception as e:
            logger.error(f"Error in enhanced query: {str(e)}")
            return []
        
    def _calculate_relevance_score(self, query: str, document: str, distance: float, metadata: Dict) -> float:
        """Calculate comprehensive relevance score using cosine similarity"""
        try:
            # Get embeddings for query and document
            query_embedding = self.embedding_function([query])[0]
            doc_embedding = self.embedding_function([document])[0]
            
            # Calculate cosine similarity using SimilarityUtils
            similarity_score = float(SimilarityUtils.cosine_similarity(
                torch.tensor(query_embedding),
                torch.tensor(doc_embedding).unsqueeze(0)
            ))
            
            # Calculate keyword overlap
            query_words = set(query.lower().split())
            doc_words = set(document.lower().split())
            keyword_overlap = len(query_words.intersection(doc_words)) / len(query_words) if query_words else 0
            
            # Consider metadata importance if available
            importance_score = float(metadata.get('importance_score', 0.5))
            
            # Weighted combination with cosine similarity
            weights = {
                'similarity': 0.6,    # Increased weight for cosine similarity
                'keyword_overlap': 0.25,
                'importance': 0.15
            }
            
            final_score = (
                similarity_score * weights['similarity'] +
                keyword_overlap * weights['keyword_overlap'] +
                importance_score * weights['importance']
            )
            
            return min(max(final_score, 0.0), 1.0)  # Ensure score is between 0 and 1
            
        except Exception as e:
            logger.error(f"Error calculating relevance score: {str(e)}")
            return 0.0

    def _enhance_query(self, query: str) -> str:
        """Enhance query with contextual understanding"""
        # Split multi-part questions
        questions = [q.strip() for q in query.split('?') if q.strip()]
        
        # Extract key entities and concepts
        doc = self.embedding_function.nlp(query)
        entities = [ent.text for ent in doc.ents]
        key_phrases = [chunk.text for chunk in doc.noun_chunks]
        
        # Combine all relevant parts
        enhanced_parts = []
        enhanced_parts.extend(questions)
        enhanced_parts.extend(entities)
        enhanced_parts.extend(key_phrases)
        
        return ' '.join(set(enhanced_parts))
    
    def get_document(self, doc_id: str) -> Optional[Dict]:
        """
        Retrieve a document and its metadata by ID, converting metadata back to original types
        """
        try:
            result = self.collection.get(
                ids=[doc_id],
                include=['documents', 'metadatas']
            )
            
            if not result['ids']:
                return None
            
            # Convert metadata back to original types
            metadata = self._convert_stored_metadata(result['metadatas'][0])
                    
            return {
                'id': doc_id,
                'text': result['documents'][0],
                'metadata': metadata
            }
            
        except Exception as e:
            logger.error(f"Error retrieving document: {str(e)}")
            raise

    def update_document_metadata(self, doc_id: str, metadata: Dict) -> bool:
        """
        Update metadata for a specific document
        
        Args:
            doc_id: Document identifier
            metadata: New metadata dictionary
            
        Returns:
            Boolean indicating success
        """
        try:
            # Get existing document
            doc = self.get_document(doc_id)
            if not doc:
                return False
            
            # Update the document with new metadata
            self.collection.update(
                ids=[doc_id],
                metadatas=[metadata]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error updating document metadata: {str(e)}")
            return False
        
    def get_document_similarities(self, query: str) -> List[Dict]:
        """
        Get detailed similarity information for debugging and analysis.
        Returns full similarity information including scores.
        """
        try:
            # Preprocess and embed query
            processed_query = self.embedding_function.preprocess_text(query)
            query_embedding = self.embedding_function([processed_query])[0]
            
            # Get collection data
            results = self.collection.get()
            
            # Check if we have data
            if not results['ids'] or not results['documents']:
                logger.info("No documents found for similarity calculation")
                return []
                
            # Verify embeddings exist
            if 'embeddings' not in results or not results['embeddings']:
                logger.error("No embeddings found in results")
                return []
                
            # Calculate similarities
            similarities = SimilarityUtils.cosine_similarity(
                query_embedding=np.array(query_embedding),
                document_embeddings=np.array(results['embeddings'])
            )
            
            # Get full ranking information
            ranked_docs = SimilarityUtils.rank_documents(
                similarities=similarities,
                documents=results['documents'],
                metadatas=results['metadatas'],
                top_k=len(results['documents']),  # Get all documents
                threshold=0.0  # No threshold to see all scores
            )
            
            return ranked_docs
            
        except Exception as e:
            logger.error(f"Error getting similarities: {str(e)}")
            return []

    # def generate_response(self,
    #     prompt: str,
    #     context: Optional[List[str]] = None,
    #     metadata_filters: Optional[Dict] = None,
    #     max_length: Optional[int] = None,
    #     temperature: Optional[float] = None) -> str:

    #     """Generate response with enhanced context integration"""
    #     try:
    #        # Get context if not provided
    #         if context is None:
    #             context = self.query_knowledge_base(
    #                 query=prompt,
    #                 filters=metadata_filters
    #             )
            
    #         context_str = "\n".join(context) if context else "No specific venue details found."

    #         # Format prompt with context
    #         formatted_prompt = self.system_prompt.format(
    #             context=context_str,
    #             question=prompt
    #         )

    #         # Process and combine context
    #         if context:
    #             # Remove duplicate information
    #             seen_content = set()
    #             unique_context = []
    #             for c in context:
    #                 if c not in seen_content:
    #                     unique_context.append(c)
    #                     seen_content.add(c)
                
    #             context_str = "\n\nRelevant Information:\n" + "\n".join(
    #                 f"- {c}" for c in unique_context
    #             )
    #         else:
    #             context_str = "No specific venue details found for this query."

    #         # Format prompt with context
    #         formatted_prompt = self.system_prompt.format(
    #             context=context_str,
    #             question=prompt
    #         )

    #         # Generate response
    #         data = {
    #             "model": self.model,
    #             "prompt": formatted_prompt,
    #             "stream": False,
    #             "temperature": temperature or settings.DEFAULT_TEMPERATURE,
    #             "max_length": max_length or settings.MAX_LENGTH
    #         }
            
    #         response = requests.post(f"{self.base_url}/generate", json=data)
            
    #         if response.status_code == 200:
    #             response_text = response.json()["response"].strip()
    #             return self._clean_response(response_text)
    #         else:
    #             raise Exception(f"Error from Ollama API: {response.text}")
            
    #     except Exception as e:
    #         logger.error(f"Error during response generation: {str(e)}")
    #         raise

    def generate_response(
        self,
        prompt: str,
        max_length: Optional[int] = None,
        temperature: Optional[float] = None
    ) -> str:
        """Generate response with enhanced logging"""
        try:
            logger.info(f"Starting generate_response with prompt: {prompt[:100]}...")
            logger.info(f"Parameters - max_length: {max_length}, temperature: {temperature}")
            
            # Get relevant context
            logger.info("Querying knowledge base...")
            context = self.query_knowledge_base(prompt)
            context_str = "\n".join(context) if context else "No specific venue details found."
            logger.info(f"Retrieved context length: {len(context_str)}")

            # Format prompt with context
            formatted_prompt = self.system_prompt.format(
                context=context_str,
                question=prompt
            )
            logger.info(f"Formatted prompt length: {len(formatted_prompt)}")

            # Generate response
            data = {
                "model": self.model,
                "prompt": formatted_prompt,
                "stream": False,
                "temperature": temperature or settings.DEFAULT_TEMPERATURE,
                "max_length": max_length or settings.MAX_LENGTH
            }
            
            logger.info(f"Sending request to Ollama API at {self.base_url}")
            response = requests.post(f"{self.base_url}/generate", json=data)
            
            # Log response status and headers
            logger.info(f"Ollama API response status: {response.status_code}")
            logger.info(f"Ollama API response headers: {dict(response.headers)}")
            
            if response.status_code == 200:
                response_json = response.json()
                logger.info("Successfully received JSON response from Ollama")
                
                if "response" not in response_json:
                    logger.error(f"Unexpected response format: {response_json}")
                    raise ValueError("Invalid response format from Ollama API")
                
                response_text = response_json["response"].strip()
                cleaned_response = response_text.replace("Assistant:", "").replace("User:", "").strip()
                logger.info(f"Final response length: {len(cleaned_response)}")
                return cleaned_response
            else:
                logger.error(f"Error response from Ollama API: {response.text}")
                raise Exception(f"Error from Ollama API: {response.text}")
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Network error with Ollama API: {str(e)}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding Ollama API response: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Error in generate_response: {str(e)}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
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
        page_size: int = 10,
        filters: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """
        List documents with enhanced filtering and pagination
        
        Args:
            search_term: Optional search string
            page: Page number
            page_size: Items per page
            filters: Optional metadata filters
            
        Returns:
            Dictionary containing paginated documents and metadata
        """
        try:
            # Prepare where clauses for filtering
            where = {}
            if filters:
                for key, value in filters.items():
                    if key in ['categories', 'content_type', 'target_audience', 'status']:
                        where[key] = value

            # Get all matching documents
            result = self.collection.get(
                where=where,
                include=['documents', 'metadatas']
            )

            documents = []
            for i, doc_id in enumerate(result['ids']):
                doc = {
                    'id': doc_id,
                    'text': result['documents'][i],
                    'metadata': result['metadatas'][i]
                }
                
                # Apply text search if provided
                if search_term and search_term.lower() not in doc['text'].lower():
                    continue
                    
                documents.append(doc)

            # Calculate pagination
            total = len(documents)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            
            return {
                'documents': documents[start_idx:end_idx],
                'total': total,
                'page': page,
                'page_size': page_size,
                'total_pages': (total + page_size - 1) // page_size
            }
            
        except Exception as e:
            logger.error(f"Error listing documents: {str(e)}")
            raise
    
    def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks"""
        try:
            # Get document to check if it's chunked
            doc = self.get_document(doc_id)
            if not doc:
                return False
                
            # If document is chunked, delete all related chunks
            if doc['metadata'].get('is_chunked'):
                base_id = doc_id.rsplit('_', 1)[0]
                # Get all chunks
                chunks = self.collection.get(
                    where={"document_id": base_id}
                )
                # Delete all chunks
                self.collection.delete(ids=chunks['ids'])
            else:
                # Delete single document
                self.collection.delete(ids=[doc_id])
                
            return True
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False

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
    
    def check_collection_status(self) -> Dict:
        """Check the status of the ChromaDB collection"""
        try:
            results = self.collection.get()
            return {
                'document_count': len(results['ids']),
                'has_embeddings': 'embeddings' in results and bool(results['embeddings']),
                'has_metadata': 'metadatas' in results and bool(results['metadatas']),
                'sample_ids': results['ids'][:5] if results['ids'] else []
            }
        except Exception as e:
            logger.error(f"Error checking collection status: {str(e)}")
            return {
                'error': str(e),
                'document_count': 0,
                'has_embeddings': False,
                'has_metadata': False,
                'sample_ids': []
            }