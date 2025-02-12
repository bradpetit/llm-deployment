# app/utils.py
import requests
import logging
from typing import List, Dict, Optional
import chromadb
from app.config import settings

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LLMManager:
    def __init__(self):
        self.base_url = "http://localhost:11434/api"
        self.model = "mistral"  # or any other model you prefer
        self.collection = None

    def initialize_models(self):
        try:
            logger.info("Starting initialization...")
            
            # Initialize ChromaDB for RAG
            logger.info("Initializing ChromaDB...")
            chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
            
            self.collection = chroma_client.get_or_create_collection(
                name="knowledge_base"
            )
            
            # Test Ollama connection
            logger.info("Testing Ollama connection...")
            response = requests.get(f"{self.base_url}/tags")
            if response.status_code == 200:
                logger.info("Successfully connected to Ollama")
            else:
                raise ConnectionError("Could not connect to Ollama service")
            
            logger.info("Initialization complete!")
            
        except Exception as e:
            logger.error(f"Error during initialization: {str(e)}")
            raise

    def generate_response(self, prompt: str, max_length: Optional[int] = None, temperature: Optional[float] = None) -> str:
        try:
            data = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            if temperature is not None:
                data["temperature"] = temperature
            
            response = requests.post(f"{self.base_url}/generate", json=data)
            
            if response.status_code == 200:
                return response.json()["response"]
            else:
                raise Exception(f"Error from Ollama API: {response.text}")
            
        except Exception as e:
            logger.error(f"Error during response generation: {str(e)}")
            raise

    def query_knowledge_base(self, query: str, n_results: int = 3) -> List[str]:
        try:
            results = self.collection.query(
                query_texts=[query],
                n_results=n_results
            )
            return results.get("documents", [[]])[0]
        except Exception as e:
            logger.error(f"Error querying knowledge base: {str(e)}")
            raise

    def add_to_knowledge_base(self, text: str, metadata: Optional[Dict] = None) -> str:
        try:
            doc_id = f"doc_{len(self.collection.get()['ids']) + 1}"
            self.collection.add(
                documents=[text],
                metadatas=[metadata or {}],
                ids=[doc_id]
            )
            return doc_id
        except Exception as e:
            logger.error(f"Error adding to knowledge base: {str(e)}")
            raise

def create_chat_prompt(conversation: List[Dict], context: str = "") -> str:
    prompt = """Below is a conversation between a user and an AI assistant. 
The assistant is helpful, knowledgeable, and friendly.

"""
    if context:
        prompt += f"Relevant context:\n{context}\n\n"
    
    prompt += "Conversation:\n"
    for message in conversation:
        role = "User" if message["role"] == "user" else "Assistant"
        prompt += f"{role}: {message['content']}\n"
    
    prompt += "Assistant:"
    return prompt