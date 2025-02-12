# app/utils.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sentence_transformers import SentenceTransformer
import chromadb
from app.config import settings

class LLMManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.embedder = None
        self.collection = None

    def initialize_models(self):
        print("Initializing models...")
        
        # Initialize LLM
        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # Initialize embedding model
        self.embedder = SentenceTransformer(settings.EMBEDDING_MODEL)
        
        # Initialize ChromaDB
        chroma_client = chromadb.PersistentClient(path=settings.CHROMA_DB_PATH)
        
        # Create or get collection
        self.collection = chroma_client.get_or_create_collection(
            name="knowledge_base",
            embedding_function=lambda texts: self.embedder.encode(texts).tolist()
        )
        
        print("Models initialized successfully")

    def generate_response(self, prompt: str, max_length: int = None, temperature: float = None):
        max_length = max_length or settings.MAX_LENGTH
        temperature = temperature or settings.DEFAULT_TEMPERATURE
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=max_length,
            temperature=temperature,
            do_sample=True,
            pad_token_id=self.tokenizer.eos_token_id
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)

    def query_knowledge_base(self, query: str, n_results: int = 3):
        results = self.collection.query(
            query_texts=[query],
            n_results=n_results
        )
        return results.get("documents", [[]])[0]

    def add_to_knowledge_base(self, text: str, metadata: dict = None):
        doc_id = f"doc_{len(self.collection.get()['ids']) + 1}"
        self.collection.add(
            documents=[text],
            metadatas=[metadata or {}],
            ids=[doc_id]
        )
        return doc_id

def create_chat_prompt(conversation: list, context: str = "") -> str:
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