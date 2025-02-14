from typing import Dict, Any, Optional, List
import re
from datetime import datetime
import hashlib
from pathlib import Path
import PyPDF2
import docx
import logging
from io import BytesIO
from sentence_transformers import SentenceTransformer

logger = logging.getLogger(__name__)

class EnhancedMetadataGenerator:
    """A comprehensive metadata generator for document processing"""
    
    def __init__(self):
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.setup_taxonomies()
        
    def setup_taxonomies(self):
        """Initialize classification taxonomies"""
        self.document_categories = {
            'pricing': ['price', 'cost', 'fee', 'payment', 'rates', 'pricing', 'package', 'discount'],
            'policies': ['policy', 'rule', 'guideline', 'requirement', 'term', 'condition', 'agreement'],
            'venue_info': ['capacity', 'space', 'facility', 'amenity', 'feature', 'location', 'area'],
            'catering': ['food', 'beverage', 'menu', 'catering', 'dining', 'meal', 'drink'],
            'event_planning': ['schedule', 'timeline', 'planning', 'coordinator', 'organization'],
            'wedding': ['ceremony', 'reception', 'bride', 'groom', 'wedding', 'marriage', 'celebration'],
            'corporate': ['meeting', 'conference', 'corporate', 'business', 'seminar', 'workshop'],
            'technical': ['equipment', 'setup', 'technical', 'audio', 'visual', 'lighting'],
            'legal': ['contract', 'agreement', 'terms', 'conditions', 'liability', 'insurance'],
            'marketing': ['promotion', 'advertising', 'brochure', 'marketing', 'social media']
        }
        
        self.content_types = {
            'guide': ['how to', 'guide', 'instruction', 'step by step', 'manual'],
            'policy': ['policy', 'rule', 'regulation', 'requirement', 'guideline'],
            'form': ['form', 'application', 'request', 'submission', 'registration'],
            'contract': ['contract', 'agreement', 'terms', 'conditions', 'legal'],
            'marketing': ['brochure', 'promotional', 'advertisement', 'marketing'],
            'informational': ['info', 'about', 'description', 'detail', 'overview']
        }
        
        self.audience_types = {
            'client': ['client', 'customer', 'guest', 'visitor'],
            'vendor': ['vendor', 'supplier', 'provider', 'partner'],
            'staff': ['staff', 'employee', 'team', 'personnel'],
            'management': ['manager', 'administrator', 'supervisor', 'director']
        }

    def generate_metadata(self, 
                         content: str, 
                         file_info: Optional[Dict] = None, 
                         existing_metadata: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Generate comprehensive metadata for a document
        
        Args:
            content: The document text content
            file_info: Optional dictionary containing file information (name, type, etc.)
            existing_metadata: Optional existing metadata to augment
            
        Returns:
            Dictionary containing generated metadata
        """
        try:
            metadata = {
                'document_id': self._generate_document_id(content),
                'timestamp': datetime.now().isoformat(),
                'processing_version': '2.0',
                
                # Content analysis
                'word_count': len(content.split()),
                'char_count': len(content),
                'language': self._detect_language(content),
                'categories': self._classify_categories(content),
                'content_type': self._classify_content_type(content),
                'target_audience': self._classify_audience(content),
                
                # Text analysis
                'keywords': self._extract_keywords(content),
                'summary': self._generate_summary(content),
                'readability_score': self._calculate_readability(content),
                'sentiment_score': self._analyze_sentiment(content),
                
                # Semantic analysis
                'topic_vector': self._generate_topic_vector(content),
                'named_entities': self._extract_entities(content),
                
                # Classification scores
                'classification_confidence': self._calculate_classification_confidence(content),
                'content_quality_score': self._assess_content_quality(content),
                
                # Technical metadata
                'version': '1.0',
                'processing_timestamp': datetime.now().isoformat(),
                'status': 'active'
            }
            
            # Add file-specific metadata if provided
            if file_info:
                metadata.update(self._extract_file_metadata(file_info))
            
            # Merge with existing metadata if provided
            if existing_metadata:
                metadata = self._merge_metadata(metadata, existing_metadata)
            
            return metadata
            
        except Exception as e:
            logger.error(f"Error generating metadata: {str(e)}")
            return self._generate_fallback_metadata(content)

    def _generate_document_id(self, content: str) -> str:
        """Generate a unique document ID based on content hash"""
        return hashlib.sha256(content.encode()).hexdigest()[:12]

    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        # This is a placeholder - you might want to use a proper language detection library
        common_english_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that'}
        words = set(text.lower().split()[:100])
        if len(words.intersection(common_english_words)) > 3:
            return 'en'
        return 'unknown'

    def _classify_categories(self, content: str) -> List[str]:
        """Classify document into multiple categories based on content"""
        content_lower = content.lower()
        categories = []
        
        for category, keywords in self.document_categories.items():
            keyword_count = sum(1 for keyword in keywords if keyword in content_lower)
            if keyword_count >= 2:  # Require at least 2 keyword matches
                categories.append(category)
        
        return categories or ['general']

    def _classify_content_type(self, content: str) -> str:
        """Determine the type of content"""
        content_lower = content.lower()
        type_scores = {}
        
        for content_type, indicators in self.content_types.items():
            score = sum(1 for indicator in indicators if indicator in content_lower)
            type_scores[content_type] = score
        
        if not type_scores:
            return 'general'
        
        return max(type_scores.items(), key=lambda x: x[1])[0]

    def _classify_audience(self, content: str) -> List[str]:
        """Identify target audience(s)"""
        content_lower = content.lower()
        audiences = []
        
        for audience, indicators in self.audience_types.items():
            if any(indicator in content_lower for indicator in indicators):
                audiences.append(audience)
        
        return audiences or ['general']

    def _extract_keywords(self, content: str, max_keywords: int = 10) -> List[str]:
        """Extract key terms from content"""
        # Simple keyword extraction based on frequency
        words = re.findall(r'\b\w+\b', content.lower())
        word_freq = {}
        
        # Common English stop words to filter out
        stop_words = {'the', 'be', 'to', 'of', 'and', 'a', 'in', 'that', 'have', 'i', 'it'}
        
        for word in words:
            if word not in stop_words and len(word) > 3:
                word_freq[word] = word_freq.get(word, 0) + 1
        
        # Get top keywords by frequency
        keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, _ in keywords[:max_keywords]]

    def _generate_summary(self, content: str, max_length: int = 200) -> str:
        """Generate a brief summary of the content"""
        # Simple extractive summarization
        sentences = content.split('.')
        if len(sentences) <= 2:
            return content[:max_length] + '...' if len(content) > max_length else content
        
        return '. '.join(sentences[:2]) + '.'

    def _calculate_readability(self, content: str) -> float:
        """Calculate readability score"""
        words = content.split()
        sentences = content.split('.')
        
        if not words or not sentences:
            return 0.0
        
        # Simple Flesch-Kincaid grade level approximation
        avg_words_per_sentence = len(words) / len(sentences)
        return round(0.39 * avg_words_per_sentence + 11.8, 1)

    def _analyze_sentiment(self, content: str) -> float:
        """Simple sentiment analysis"""
        # This is a very basic implementation
        positive_words = {'good', 'great', 'excellent', 'amazing', 'wonderful', 'best', 'perfect'}
        negative_words = {'bad', 'poor', 'terrible', 'worst', 'awful', 'horrible'}
        
        words = content.lower().split()
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        if not (positive_count + negative_count):
            return 0.0
            
        return round((positive_count - negative_count) / (positive_count + negative_count), 2)

    def _generate_topic_vector(self, content: str) -> List[float]:
        """Generate topic embedding vector"""
        try:
            # Generate embedding for the first 512 tokens
            embedding = self.embedding_model.encode(content[:1024])
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating topic vector: {str(e)}")
            return []

    def _extract_entities(self, content: str) -> Dict[str, List[str]]:
        """Extract named entities"""
        entities = {
            'dates': [],
            'amounts': [],
            'contacts': []
        }
        
        # Extract dates (simple regex for demonstration)
        date_pattern = r'\d{1,2}/\d{1,2}/\d{2,4}'
        entities['dates'] = re.findall(date_pattern, content)
        
        # Extract monetary amounts
        amount_pattern = r'\$\d+(?:,\d{3})*(?:\.\d{2})?'
        entities['amounts'] = re.findall(amount_pattern, content)
        
        # Extract email addresses
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        entities['contacts'] = re.findall(email_pattern, content)
        
        return entities

    def _calculate_classification_confidence(self, content: str) -> float:
        """Calculate confidence score for classifications"""
        # Simple confidence calculation based on keyword matches
        total_keywords = sum(len(keywords) for keywords in self.document_categories.values())
        matched_keywords = sum(
            1 for keywords in self.document_categories.values()
            for keyword in keywords if keyword in content.lower()
        )
        
        return round(matched_keywords / total_keywords, 2)

    def _assess_content_quality(self, content: str) -> float:
        """Assess overall content quality"""
        factors = {
            'length': min(1.0, len(content) / 1000),  # Normalize to 1000 chars
            'formatting': 0.5 + (0.5 * ('.' in content and ',' in content)),  # Basic punctuation
            'structure': 0.5 + (0.5 * ('\n' in content))  # Basic structure check
        }
        
        return round(sum(factors.values()) / len(factors), 2)

    def _extract_file_metadata(self, file_info: Dict) -> Dict[str, Any]:
        """Extract metadata specific to file type"""
        metadata = {
            'filename': file_info.get('filename', ''),
            'file_type': file_info.get('content_type', ''),
            'file_size': file_info.get('size', 0)
        }
        
        # Add file type specific metadata
        if file_info.get('content_type') == 'application/pdf':
            metadata.update(self._extract_pdf_metadata(file_info.get('content', b'')))
        elif file_info.get('content_type') == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            metadata.update(self._extract_docx_metadata(file_info.get('content', b'')))
            
        return metadata

    def _extract_pdf_metadata(self, content: bytes) -> Dict[str, Any]:
        """Extract PDF-specific metadata"""
        try:
            pdf = PyPDF2.PdfReader(BytesIO(content))
            return {
                'page_count': len(pdf.pages),
                'pdf_version': pdf.pdf_header,
                'is_encrypted': pdf.is_encrypted,
                'author': pdf.metadata.get('/Author', ''),
                'creation_date': pdf.metadata.get('/CreationDate', '')
            }
        except Exception as e:
            logger.error(f"Error extracting PDF metadata: {str(e)}")
            return {}

    def _extract_docx_metadata(self, content: bytes) -> Dict[str, Any]:
        """Extract DOCX-specific metadata"""
        try:
            doc = docx.Document(BytesIO(content))
            core_props = doc.core_properties
            return {
                'author': core_props.author or '',
                'created': str(core_props.created) if core_props.created else '',
                'last_modified_by': core_props.last_modified_by or '',
                'revision': core_props.revision,
                'paragraph_count': len(doc.paragraphs)
            }
        except Exception as e:
            logger.error(f"Error extracting DOCX metadata: {str(e)}")
            return {}

    def _merge_metadata(self, new_metadata: Dict, existing_metadata: Dict) -> Dict:
        """Merge new metadata with existing metadata"""
        merged = existing_metadata.copy()
        
        # Update existing fields with new values
        for key, value in new_metadata.items():
            if key in merged:
                if isinstance(merged[key], list) and isinstance(value, list):
                    # Combine lists without duplicates
                    merged[key] = list(set(merged[key] + value))
                elif isinstance(merged[key], dict) and isinstance(value, dict):
                    # Deep merge dictionaries
                    merged[key].update(value)
                else:
                    # Override with new value
                    merged[key] = value
            else:
                # Add new field
                merged[key] = value
        
        return merged

    def _generate_fallback_metadata(self, content: str) -> Dict[str, Any]:
        """Generate basic metadata when full processing fails"""
        return {
            'document_id': self._generate_document_id(content),
            'timestamp': datetime.now().isoformat(),
            'word_count': len(content.split()),
            'char_count': len(content),
            'status': 'partial',
            'processing_error': True,
            'processing_version': '2.0'
        }

    def update_metadata(self, 
                       content: str, 
                       existing_metadata: Dict, 
                       update_fields: List[str] = None) -> Dict[str, Any]:
        """
        Update specific metadata fields while preserving others
        
        Args:
            content: The document content
            existing_metadata: Existing metadata to update
            update_fields: List of specific fields to update (None for all)
            
        Returns:
            Updated metadata dictionary
        """
        try:
            # Generate new metadata
            new_metadata = self.generate_metadata(content)
            
            if update_fields:
                # Update only specified fields
                updated = existing_metadata.copy()
                for field in update_fields:
                    if field in new_metadata:
                        updated[field] = new_metadata[field]
            else:
                # Full merge of new and existing metadata
                updated = self._merge_metadata(new_metadata, existing_metadata)
            
            # Add update tracking
            updated['last_updated'] = datetime.now().isoformat()
            updated['update_count'] = existing_metadata.get('update_count', 0) + 1
            
            return updated
            
        except Exception as e:
            logger.error(f"Error updating metadata: {str(e)}")
            return existing_metadata

    def batch_process(self, documents: List[Dict]) -> List[Dict[str, Any]]:
        """
        Process multiple documents in batch
        
        Args:
            documents: List of dictionaries containing document content and optional file info
            
        Returns:
            List of generated metadata dictionaries
        """
        results = []
        for doc in documents:
            try:
                metadata = self.generate_metadata(
                    content=doc['content'],
                    file_info=doc.get('file_info'),
                    existing_metadata=doc.get('existing_metadata')
                )
                results.append(metadata)
            except Exception as e:
                logger.error(f"Error processing document in batch: {str(e)}")
                results.append(self._generate_fallback_metadata(doc['content']))
        
        return results

    def validate_metadata(self, metadata: Dict) -> Dict[str, List[str]]:
        """
        Validate metadata fields and return any issues
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Dictionary of validation issues by field
        """
        issues = {}
        
        # Required fields
        required_fields = ['document_id', 'timestamp', 'word_count']
        for field in required_fields:
            if field not in metadata:
                issues[field] = [f"Missing required field: {field}"]
        
        # Type validation
        if 'word_count' in metadata and not isinstance(metadata['word_count'], int):
            issues['word_count'] = ["word_count must be an integer"]
            
        if 'categories' in metadata and not isinstance(metadata['categories'], list):
            issues['categories'] = ["categories must be a list"]
            
        if 'topic_vector' in metadata and not isinstance(metadata['topic_vector'], list):
            issues['topic_vector'] = ["topic_vector must be a list of floats"]
        
        # Value validation
        if 'readability_score' in metadata:
            score = metadata['readability_score']
            if not isinstance(score, (int, float)) or score < 0:
                issues['readability_score'] = ["readability_score must be a non-negative number"]
        
        if 'sentiment_score' in metadata:
            score = metadata['sentiment_score']
            if not isinstance(score, (int, float)) or score < -1 or score > 1:
                issues['sentiment_score'] = ["sentiment_score must be between -1 and 1"]
        
        return issues

    def export_metadata_summary(self, metadata: Dict) -> Dict[str, Any]:
        """
        Create a human-readable summary of metadata
        
        Args:
            metadata: Full metadata dictionary
            
        Returns:
            Dictionary with summarized metadata
        """
        return {
            'document_id': metadata.get('document_id', ''),
            'categories': ', '.join(metadata.get('categories', [])),
            'content_type': metadata.get('content_type', 'unknown'),
            'word_count': metadata.get('word_count', 0),
            'summary': metadata.get('summary', ''),
            'keywords': ', '.join(metadata.get('keywords', []))[:100],
            'last_updated': metadata.get('last_updated', metadata.get('timestamp', '')),
            'quality_score': metadata.get('content_quality_score', 0.0),
            'status': metadata.get('status', 'unknown')
        }