from enum import Enum
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from datetime import datetime

class TicketCategory(Enum):
    TECHNICAL = "technical"
    BILLING = "billing"
    FEATURE = "feature"
    ACCESS = "access"

class Priority(Enum):
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    URGENT = 4

@dataclass
class TicketAnalysis:
    category: TicketCategory
    priority: Priority
    key_points: List[str]
    required_expertise: List[str]
    sentiment: float
    urgency_indicators: List[str]
    business_impact: str
    suggested_response_type: str

@dataclass
class ResponseSuggestion:
    response_text: str
    confidence_score: float
    requires_approval: bool
    suggested_actions: List[str]

@dataclass
class SupportTicket:
    id: str
    subject: str
    content: str
    customer_info: Dict[str, Any]

@dataclass
class TicketResolution:
    ticket_id: str
    analysis: Optional[TicketAnalysis]
    response: Optional[ResponseSuggestion]
    processed_at: datetime
    processing_time: float
    status: str
    error: Optional[str] = None