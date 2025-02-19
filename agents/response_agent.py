from typing import Dict, Any
import json
from openai import OpenAI
from .data_types import ResponseSuggestion, TicketAnalysis

class ResponseAgent:
    def __init__(self, openai_api_key: str):
        self.client = OpenAI(api_key=openai_api_key)

    async def generate_response(
        self,
        ticket_analysis: TicketAnalysis,
        response_templates: Dict[str, str],
        context: Dict[str, Any]
    ) -> ResponseSuggestion:
        # Define the response generation prompt with function calling
        tools = [{
            "type": "function",
            "function": {
                "name": "generate_support_response",
                "description": "Generate a response for a support ticket",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "response_text": {
                            "type": "string",
                            "description": "The generated response text"
                        },
                        "confidence_score": {
                            "type": "number",
                            "description": "Confidence score between 0 and 1"
                        },
                        "requires_approval": {
                            "type": "boolean",
                            "description": "Whether this response needs human approval"
                        },
                        "suggested_actions": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of suggested follow-up actions"
                        }
                    },
                    "required": ["response_text", "confidence_score", "requires_approval", "suggested_actions"],
                    "additionalProperties": False
                },
                "strict": True
            }
        }]

        # Prepare the system message with templates and context
        system_message = f"""You are an expert support response generator. Generate a response based on:
        1. The ticket analysis provided
        2. Available response templates
        3. Context information
        
        Available templates:
        {json.dumps(response_templates, indent=2)}
        
        Guidelines:
        - Use appropriate template based on ticket category
        - Personalize the response using context
        - Match technical detail level to customer expertise
        - Include clear action items
        - Mark for approval if response involves sensitive issues"""

        # Create messages
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"""
            Generate a response for this ticket analysis:
            Category: {ticket_analysis.category.value}
            Priority: {ticket_analysis.priority.value}
            Key Points: {ticket_analysis.key_points}
            Sentiment: {ticket_analysis.sentiment}
            Business Impact: {ticket_analysis.business_impact}
            
            Context Information:
            {json.dumps(context, indent=2)}
            """}
        ]

        # Get the response from the model
        response = self.client.chat.completions.create(
            model="gpt-4-turbo-preview",
            messages=messages,
            tools=tools,
            tool_choice={"type": "function", "function": {"name": "generate_support_response"}}
        )

        # Parse the function call response
        function_call = response.choices[0].message.tool_calls[0]
        response_result = json.loads(function_call.function.arguments)

        # Create and return the ResponseSuggestion object
        return ResponseSuggestion(
            response_text=response_result["response_text"],
            confidence_score=response_result["confidence_score"],
            requires_approval=response_result["requires_approval"],
            suggested_actions=response_result["suggested_actions"]
        )