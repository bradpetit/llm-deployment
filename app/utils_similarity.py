import torch
import numpy as np

from typing import Dict, List, Tuple, Union
from torch.nn import functional as F

class SimilarityUtils:
    @staticmethod
    def cosine_similarity(
        query_embedding: Union[torch.Tensor, np.ndarray],
        document_embeddings: Union[torch.Tensor, np.ndarray]
    ) -> torch.Tensor:
        """
        Calculate cosine similarity between query and document embeddings.
        
        Args:
            query_embedding: Single query embedding vector
            document_embeddings: Matrix of document embeddings
            
        Returns:
            Tensor of similarity scores for each document
        """
        # Convert to torch tensors if needed
        if isinstance(query_embedding, np.ndarray):
            query_embedding = torch.from_numpy(query_embedding)
        if isinstance(document_embeddings, np.ndarray):
            document_embeddings = torch.from_numpy(document_embeddings)
        
        # Ensure query_embedding is 2D
        if query_embedding.dim() == 1:
            query_embedding = query_embedding.unsqueeze(0)
        
        # Normalize embeddings
        query_embedding = F.normalize(query_embedding, p=2, dim=1)
        document_embeddings = F.normalize(document_embeddings, p=2, dim=1)
        
        # Calculate cosine similarity
        similarities = torch.mm(query_embedding, document_embeddings.t())
        return similarities.squeeze()

    @staticmethod
    def rank_documents(
        similarities: torch.Tensor,
        documents: List[str],
        metadatas: List[Dict],
        top_k: int = 5,
        threshold: float = 0.5,
        similarity_weight: float = 0.7,
        importance_weight: float = 0.3
    ) -> List[Dict]:
        """
        Rank documents based on similarities and metadata importance.
        
        Args:
            similarities: Tensor of similarity scores
            documents: List of document texts
            metadatas: List of document metadata dictionaries
            top_k: Number of top documents to return
            threshold: Minimum score threshold
            similarity_weight: Weight for similarity score in final ranking
            importance_weight: Weight for importance score in final ranking
            
        Returns:
            List of dictionaries containing ranked documents with scores
        """
        # Convert similarities to numpy for easier handling
        similarities = similarities.cpu().numpy()
        
        # Combine with metadata importance scores
        ranked_results = []
        for i, (sim_score, doc, meta) in enumerate(zip(similarities, documents, metadatas)):
            # Get importance score from metadata if available
            importance = meta.get('importance_score', 0.5)
            
            # Calculate final score (weighted combination)
            final_score = (sim_score * similarity_weight) + (importance * importance_weight)
            
            # Only include if above threshold
            if final_score >= threshold:
                ranked_results.append({
                    'text': doc,
                    'similarity': float(sim_score),
                    'importance': importance,
                    'final_score': float(final_score),
                    'metadata': meta
                })
        
        # Sort by final score and get top_k
        ranked_results.sort(key=lambda x: x['final_score'], reverse=True)
        return ranked_results[:top_k]
    
    @staticmethod
    def get_similar_chunks(
        query_embedding: Union[torch.Tensor, np.ndarray],
        chunk_embeddings: Union[torch.Tensor, np.ndarray],
        chunks: List[str],
        min_similarity: float = 0.5
    ) -> List[Tuple[str, float]]:
        """
        Find similar chunks based on cosine similarity.
        
        Args:
            query_embedding: Query embedding vector
            chunk_embeddings: Matrix of chunk embeddings
            chunks: List of text chunks
            min_similarity: Minimum similarity threshold
            
        Returns:
            List of tuples containing (chunk_text, similarity_score)
        """
        # Calculate similarities
        similarities = SimilarityUtils.cosine_similarity(query_embedding, chunk_embeddings)
        
        # Convert to numpy
        similarities = similarities.cpu().numpy()
        
        # Filter and sort results
        similar_chunks = []
        for chunk, score in zip(chunks, similarities):
            if score >= min_similarity:
                similar_chunks.append((chunk, float(score)))
        
        # Sort by similarity score
        similar_chunks.sort(key=lambda x: x[1], reverse=True)
        return similar_chunks