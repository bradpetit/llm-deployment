import re
import spacy
import logging
import hashlib
import numpy as np
import torch
from torch.nn import functional as F
from typing import List
from sentence_transformers import SentenceTransformer
from .text_chunking import EnhancedChunker

logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

class EnhancedEmbedder:
    def __init__(self):

        self.models = {
            'mpnet': SentenceTransformer('all-mpnet-base-v2'),
            'minilm': SentenceTransformer('all-MiniLM-L12-v2'),
            'multiqa': SentenceTransformer('multi-qa-mpnet-base-dot-v1')
        }

        self.chunker = EnhancedChunker()
        
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
        
        self.chunk_size = 512
        self.chunk_overlap = 50
        
        # Cache for embeddings
        self.embedding_cache = {}
        self.cache_size = 10000  # Maximum number of cached embeddings

    def __call__(self, input: List[str]) -> List[List[float]]:
        """ChromaDB-compatible embedding function with ensemble approach"""
        all_embeddings = []
        
        for text in input:
            try:
                # Check cache first
                cache_key = self._generate_cache_key(text)
                if cache_key in self.embedding_cache:
                    all_embeddings.append(self.embedding_cache[cache_key])
                    continue

                # Preprocess text
                processed_text = self._preprocess_text(text)
                
                # Use only mpnet model for stable dimensionality
                embedding = self.models['mpnet'].encode(processed_text)
                
                # Apply importance weighting
                importance_score = self._calculate_importance(processed_text)
                final_embedding = embedding * importance_score
                
                # Normalize the final embedding
                normalized_embedding = self._normalize_embedding(final_embedding)
                
                # Cache the result
                self._cache_embedding(cache_key, normalized_embedding)
                
                all_embeddings.append(normalized_embedding)
                
            except Exception as e:
                logger.error(f"Error processing text: {str(e)}")
                # Fallback to basic embedding
                fallback_embedding = self.models['mpnet'].encode(text)
                normalized_fallback = self._normalize_embedding(fallback_embedding)
                all_embeddings.append(normalized_fallback)
        
        # Ensure all embeddings have the same dimension
        return all_embeddings

    def _generate_cache_key(self, text: str) -> str:
        """Generate a cache key for the text"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def _cache_embedding(self, key: str, embedding: np.ndarray):
        """Cache embedding with LRU strategy"""
        if len(self.embedding_cache) >= self.cache_size:
            # Remove oldest item
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        self.embedding_cache[key] = embedding
    
    def _preprocess_text(self, text: str) -> str:
        """Enhanced text preprocessing"""
        # Basic cleaning
        text = text.strip()
        text = re.sub(r'\s+', ' ', text)
        
        # Process with spaCy
        doc = self.nlp(text)
        
        # Extract key phrases and entities
        key_phrases = []
        
        # Add named entities with their labels
        for ent in doc.ents:
            key_phrases.append(f"{ent.label_}: {ent.text}")
        
        # Add noun phrases
        for chunk in doc.noun_chunks:
            key_phrases.append(chunk.text)
        
        # Add main verbs with their subjects and objects
        for token in doc:
            if token.dep_ == "ROOT" and token.pos_ == "VERB":
                verb_phrase = []
                for child in token.children:
                    if child.dep_ in ["nsubj", "dobj"]:
                        verb_phrase.append(child.text)
                if verb_phrase:
                    key_phrases.append(f"{' '.join(verb_phrase)} {token.text}")
        
        # Combine original text with key phrases
        enhanced_text = f"{text} | {' | '.join(key_phrases)}"
        return enhanced_text
    
    def _combine_embeddings(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """Combine multiple embeddings with attention mechanism"""
        try:
            # Convert all embeddings to the same dimension (use smallest dimension)
            min_dim = min(emb.shape[0] for emb in embeddings)
            resized_embeddings = [emb[:min_dim] for emb in embeddings]
            
            # Stack embeddings
            stacked = np.stack(resized_embeddings)
            
            # Calculate attention scores using dot product
            attention_scores = np.matmul(stacked, stacked.T)
            
            # Convert to PyTorch tensor for softmax
            attention_weights = F.softmax(torch.from_numpy(attention_scores).float(), dim=-1).numpy()
            
            # Reshape attention weights to match stacked embeddings
            attention_weights = attention_weights.reshape(-1, 1)
            
            # Ensure dimensions match for multiplication
            weighted_embeddings = stacked * attention_weights
            
            # Sum along the first axis
            combined = np.sum(weighted_embeddings, axis=0)
            
            return combined
            
        except Exception as e:
            logger.error(f"Error combining embeddings: {str(e)}")
            # Fallback to simple averaging
            return np.mean(embeddings, axis=0)
    
    def _normalize_embedding(self, embedding: np.ndarray) -> List[float]:
        """Normalize embedding vector"""
        norm = np.linalg.norm(embedding)
        if norm > 0:
            normalized = embedding / norm
        else:
            normalized = embedding
        return normalized.tolist()
    
    def _calculate_importance(self, text: str) -> float:
        """Calculate semantic importance score"""
        doc = self.nlp(text)
        
        # Calculate various importance factors
        entity_score = len(doc.ents) * 0.3
        noun_chunks_score = len(list(doc.noun_chunks)) * 0.2
        
        # Calculate semantic density
        total_tokens = len(doc)
        content_tokens = sum(1 for token in doc if not token.is_stop and not token.is_punct)
        semantic_density = (content_tokens / total_tokens) if total_tokens > 0 else 0
        
        # Calculate verb complexity
        verb_complexity = sum(1 for token in doc if token.pos_ == "VERB" and token.dep_ in ["ROOT", "ccomp", "xcomp"])
        
        # Combine scores
        total_score = entity_score + noun_chunks_score + semantic_density + (verb_complexity * 0.1)
        
        # Normalize to range [0.5, 1.5] to avoid extreme scaling
        normalized_score = 0.5 + (min(total_score, 10) / 10)
        
        return normalized_score
    
    def create_semantic_chunks(self, text: str) -> List[str]:
        """Create semantic chunks from text"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_size = 0
        
        for sent in doc.sents:
            sent_size = len(sent.text.split())
            
            if current_size + sent_size <= self.chunk_size:
                current_chunk.append(sent.text)
                current_size += sent_size
            else:
                if current_chunk:
                    chunks.append(" ".join(current_chunk))
                current_chunk = [sent.text]
                current_size = sent_size
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
        return chunks if chunks else [text]

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
