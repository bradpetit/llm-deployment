from typing import List, Dict
import spacy
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class TextChunk:
    """Represents a chunk of text with metadata"""
    text: str
    importance_score: float
    semantic_density: float
    source_location: Dict[str, int]  # start/end positions
    key_entities: List[str]

class EnhancedChunker:
    def __init__(self):
        # Load spaCy model
        try:
            self.nlp = spacy.load('en_core_web_sm')
        except OSError:
            logger.info("Downloading spaCy model...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load('en_core_web_sm')
        
        # Chunking parameters
        self.min_chunk_size = 100
        self.max_chunk_size = 512
        self.overlap = 50
        self.min_sentence_length = 10

    def create_chunks(self, text: str, strategy: str = 'semantic') -> List[TextChunk]:
        """
        Create chunks using specified strategy
        
        Args:
            text: Input text to chunk
            strategy: Chunking strategy ('semantic', 'fixed', or 'hybrid')
            
        Returns:
            List of TextChunk objects
        """
        if strategy == 'semantic':
            return self._semantic_chunking(text)
        elif strategy == 'fixed':
            return self._fixed_size_chunking(text)
        else:  # hybrid
            return self._hybrid_chunking(text)

    def _semantic_chunking(self, text: str) -> List[TextChunk]:
        """Split text based on semantic boundaries"""
        doc = self.nlp(text)
        chunks = []
        current_chunk = []
        current_length = 0
        chunk_start = 0
        
        # Process each sentence
        for sent in doc.sents:
            sent_text = sent.text.strip()
            sent_length = len(sent_text.split())
            
            # Skip very short sentences
            if sent_length < self.min_sentence_length:
                continue
            
            # Check if adding this sentence exceeds max chunk size
            if current_length + sent_length > self.max_chunk_size and current_chunk:
                # Create chunk from accumulated sentences
                chunk_text = ' '.join(current_chunk)
                chunk = self._create_chunk_with_metadata(
                    text=chunk_text,
                    start_pos=chunk_start,
                    end_pos=sent.start_char
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                overlap_tokens = current_chunk[-2:] if len(current_chunk) > 2 else current_chunk[-1:]
                current_chunk = overlap_tokens + [sent_text]
                current_length = sum(len(t.split()) for t in current_chunk)
                chunk_start = doc[doc.char_span(chunk_start).start].sent.start_char
            else:
                current_chunk.append(sent_text)
                current_length += sent_length
        
        # Add final chunk if there's remaining text
        if current_chunk:
            chunk_text = ' '.join(current_chunk)
            chunk = self._create_chunk_with_metadata(
                text=chunk_text,
                start_pos=chunk_start,
                end_pos=len(text)
            )
            chunks.append(chunk)
        
        return chunks

    def _fixed_size_chunking(self, text: str) -> List[TextChunk]:
        """Split text into fixed-size chunks with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), self.max_chunk_size - self.overlap):
            chunk_words = words[i:i + self.max_chunk_size]
            if len(chunk_words) < self.min_chunk_size and i > 0:
                # If chunk is too small, merge with previous chunk
                continue
                
            chunk_text = ' '.join(chunk_words)
            start_pos = len(' '.join(words[:i])) + (1 if i > 0 else 0)
            end_pos = start_pos + len(chunk_text)
            
            chunk = self._create_chunk_with_metadata(
                text=chunk_text,
                start_pos=start_pos,
                end_pos=end_pos
            )
            chunks.append(chunk)
            
        return chunks

    def _hybrid_chunking(self, text: str) -> List[TextChunk]:
        """Combine semantic and fixed-size chunking strategies"""
        # First try semantic chunking
        semantic_chunks = self._semantic_chunking(text)
        
        # Check if any chunks exceed max size
        final_chunks = []
        for chunk in semantic_chunks:
            if len(chunk.text.split()) > self.max_chunk_size:
                # Split oversized chunks using fixed-size strategy
                sub_chunks = self._fixed_size_chunking(chunk.text)
                final_chunks.extend(sub_chunks)
            else:
                final_chunks.append(chunk)
        
        return final_chunks

    def _create_chunk_with_metadata(self, text: str, start_pos: int, end_pos: int) -> TextChunk:
        """Create a TextChunk with metadata"""
        doc = self.nlp(text)
        
        # Extract key entities
        key_entities = [ent.text for ent in doc.ents]
        
        # Calculate semantic density (entities and key phrases per token)
        token_count = len(doc)
        entity_count = len(doc.ents)
        noun_chunks = len(list(doc.noun_chunks))
        semantic_density = (entity_count + noun_chunks) / token_count if token_count > 0 else 0
        
        # Calculate importance score
        importance_score = self._calculate_importance_score(doc)
        
        return TextChunk(
            text=text,
            importance_score=importance_score,
            semantic_density=semantic_density,
            source_location={'start': start_pos, 'end': end_pos},
            key_entities=key_entities
        )

    def _calculate_importance_score(self, doc: spacy.tokens.Doc) -> float:
        """Calculate chunk importance score based on multiple factors"""
        # Entity presence
        entity_score = len(doc.ents) * 0.3
        
        # Key phrase density
        noun_chunk_score = len(list(doc.noun_chunks)) * 0.2
        
        # Important keywords
        keyword_score = sum(1 for token in doc 
                          if not token.is_stop 
                          and token.pos_ in ['NOUN', 'VERB', 'ADJ']) * 0.1
        
        # Sentence structure complexity
        complexity_score = sum(1 for token in doc 
                             if token.dep_ in ['ccomp', 'xcomp', 'advcl']) * 0.2
        
        # Normalize final score to 0-1 range
        total_score = entity_score + noun_chunk_score + keyword_score + complexity_score
        normalized_score = min(total_score / 10, 1.0)  # Normalize assuming max score of 10
        
        return normalized_score

    def merge_small_chunks(self, chunks: List[TextChunk], min_size: int = 100) -> List[TextChunk]:
        """Merge chunks that are too small"""
        if not chunks:
            return chunks
            
        merged = []
        current_chunk = chunks[0]
        
        for next_chunk in chunks[1:]:
            current_size = len(current_chunk.text.split())
            if current_size < min_size:
                # Merge with next chunk
                merged_text = current_chunk.text + ' ' + next_chunk.text
                current_chunk = self._create_chunk_with_metadata(
                    text=merged_text,
                    start_pos=current_chunk.source_location['start'],
                    end_pos=next_chunk.source_location['end']
                )
            else:
                merged.append(current_chunk)
                current_chunk = next_chunk
        
        # Add final chunk
        merged.append(current_chunk)
        return merged