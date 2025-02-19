from typing import Optional, Dict, Any
import json
from openai import OpenAI
from huggingface_hub import InferenceClient
from .data_types import TicketAnalysis, TicketCategory, Priority

class TicketAnalysisAgent:
    def __init__(self, openai_api_key: str, hf_api_key: str):
        self.openai_client = OpenAI(api_key=openai_api_key)
        self.hf_client = InferenceClient(token=hf_api_key)
        
    async def analyze_ticket(
        self,
        ticket_content: str,
        customer_info: Optional[Dict[str, Any]] = None
    ) -> TicketAnalysis:
        # Define the analysis prompt with function calling
        tools = [{
            "type": "function",
            "function": {
                "name": "analyze_support_ticket",
                "description": "Analyze a customer support ticket for classification and priority",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "category": {
                            "type": "string",
                            "enum": ["technical", "billing", "feature", "access"],
                            "description": "The category of the support ticket"
                        },
                        "priority": {
                            "type": "integer",
                            "enum": [1, 2, 3, 4],
                            "description": "Priority level (1=LOW, 2=MEDIUM, 3=HIGH, 4=URGENT)"
                        },
                        "key_points": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Main points from the ticket"
                        },
                        "required_expertise": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Required expertise to handle this ticket"
                        },
                        "urgency_indicators": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Words/phrases indicating urgency"
                        },
                        "business_impact": {
                            "type": "string",
                            "description": "Description of business impact"
                        },
                        "suggested_response_type": {
                            "type": "string",
                            "description": "Suggested type of response template"
                        }
                    },
                    "required": ["category", "priority", "key_points", "required_expertise", 
                               "urgency_indicators", "business_impact", "suggested_response_type"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

        # Prepare the system message
        system_message = """You are an expert support ticket analyzer. Analyze the ticket based on:
        1. Content and subject for category
        2. Priority based on:
           - Urgency words ("ASAP", "urgent", "immediately")
           - Customer role (C-level, Director gets higher priority)
           - Business impact (payroll, revenue-impacting issues)
        3. Extract key points for response
        4. Identify required expertise
        5. Analyze business impact"""

        # Create messages including customer info if available
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"Analyze this support ticket: {ticket_content}\n\nCustomer Info: {json.dumps(customer_info) if customer_info else 'None'}"}
        ]

        # Get the analysis from the model
        response = self.openai_client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "analyze_support_ticket"}}
        )

        # Parse the function call response
        function_call = response.choices[0].message.tool_calls[0]
        analysis_result = json.loads(function_call.function.arguments)

        # Get sentiment using working Hugging Face implementation
        sentiment = await self._analyze_sentiment(ticket_content)

        # Create and return the TicketAnalysis object
        return TicketAnalysis(
            category=TicketCategory(analysis_result["category"]),
            priority=Priority(analysis_result["priority"]),
            key_points=analysis_result["key_points"],
            required_expertise=analysis_result["required_expertise"],
            sentiment=sentiment,
            urgency_indicators=analysis_result["urgency_indicators"],
            business_impact=analysis_result["business_impact"],
            suggested_response_type=analysis_result["suggested_response_type"]
        )

    async def _analyze_sentiment(self, text: str) -> float:
        try:
            # Make API call for similarity computation
            response = self.hf_client.post(
                json={
                    "inputs": {
                        "source_sentence": text,
                        "sentences": [
                            "I am very happy and satisfied with the service",
                            "I am very frustrated and unhappy with the service"
                        ]
                    }
                },
                model="sentence-transformers/all-MiniLM-L6-v2",
                task="sentence-similarity"
            )
            
            # Parse the response
            similarities = json.loads(response)
            
            # Calculate sentiment (positive similarity minus negative similarity)
            sentiment_score = similarities[0] - similarities[1]
            
            return sentiment_score
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            return 0.0  # Fallback to neutral sentiment