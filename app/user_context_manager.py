import re
import logging
from typing import Optional, Dict
from datetime import datetime
from app.models import UserContext

logger = logging.getLogger(__name__)

class UserContextManager:
    def __init__(self):
        self.user_contexts: Dict[str, UserContext] = {}
        
        # Patterns for extracting user information
        self.patterns = {
            'name': r'(?i)(?:my name is|i\'m|i am|call me) ([A-Za-z\s]+)',
            'email': r'[\w\.-]+@[\w\.-]+\.\w+',
            'phone': r'(?:\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}',
            'date_preference': r'(?i)(?:prefer|want|looking at|interested in) (?:the date|) ?(\d{1,2}(?:st|nd|rd|th)? (?:of )?\w+ \d{4}|\w+ \d{1,2}(?:st|nd|rd|th)?,? \d{4})',
            'guest_count': r'(?i)(?:with|for|planning for|expecting) (\d+) (?:people|guests|attendees)',
            'event_type': r'(?i)(?:planning|organizing|having) (?:a |an )?(\w+ (?:wedding|party|ceremony|reception|event|celebration))'
        }

    def get_or_create_context(self, user_id: str) -> UserContext:
        """Get existing context or create new one"""
        if user_id not in self.user_contexts:
            self.user_contexts[user_id] = UserContext(
                user_id=user_id,
                conversation_history=[],
                extracted_details={},
                last_interaction=datetime.now()
            )
        return self.user_contexts[user_id]

    def update_context(self, user_id: str, message_content: str) -> UserContext:
        """Update user context with new information from message"""
        context = self.get_or_create_context(user_id)
        
        # Extract information from message
        extracted_info = {}
        for key, pattern in self.patterns.items():
            matches = re.findall(pattern, message_content)
            if matches:
                extracted_info[key] = matches[0].strip()
        
        # Update context with new information
        if extracted_info:
            context.extracted_details.update(extracted_info)
            
            # Update specific fields if found
            if 'name' in extracted_info and not context.name:
                context.name = extracted_info['name']
            if 'email' in extracted_info and not context.email:
                context.email = extracted_info['email']
            if 'phone' in extracted_info and not context.phone:
                context.phone = extracted_info['phone']
        
        # Update last interaction time
        context.last_interaction = datetime.now()
        
        # Add message to conversation history
        context.conversation_history.append({
            'content': message_content,
            'timestamp': datetime.now().isoformat(),
            'extracted_info': extracted_info
        })
        
        # Keep only last 10 messages in history
        if len(context.conversation_history) > 10:
            context.conversation_history = context.conversation_history[-10:]
        
        return context

    def format_context_for_prompt(self, context: UserContext) -> str:
        """Format user context for inclusion in LLM prompt"""
        prompt_parts = ["Previous context:"]
        
        if context.name:
            prompt_parts.append(f"User's name: {context.name}")
        
        if context.extracted_details:
            prompt_parts.append("\nKnown details:")
            for key, value in context.extracted_details.items():
                if key not in ['name', 'email', 'phone']:  # Skip personal info
                    prompt_parts.append(f"- {key.replace('_', ' ').title()}: {value}")
        
        if context.conversation_history:
            prompt_parts.append("\nRecent conversation points:")
            for msg in context.conversation_history[-3:]:  # Last 3 messages
                if 'extracted_info' in msg and msg['extracted_info']:
                    points = [f"- {k}: {v}" for k, v in msg['extracted_info'].items()]
                    if points:
                        prompt_parts.extend(points)
        
        return "\n".join(prompt_parts)

    def clear_context(self, user_id: str):
        """Clear user context"""
        if user_id in self.user_contexts:
            del self.user_contexts[user_id]