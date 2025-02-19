import logging
import json
import asyncio
from datetime import datetime
from typing import List, Dict, Any
from .ticket_analysis import TicketAnalysisAgent
from .response_agent import ResponseAgent
from .data_types import (
    SupportTicket,
    TicketResolution,
    TicketAnalysis,
    ResponseSuggestion
)

class TicketProcessor:
    def __init__(self, openai_api_key: str, hf_api_key: str):
        self.analysis_agent = TicketAnalysisAgent(openai_api_key, hf_api_key)
        self.response_agent = ResponseAgent(openai_api_key)
        self.context = {}
        self.logger = logging.getLogger(__name__)
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    async def process_ticket(
        self,
        ticket: SupportTicket,
    ) -> TicketResolution:
        start_time = datetime.now()
        
        try:
            # 1. Analyze the ticket
            self.logger.info(f"Starting analysis for ticket {ticket.id}")
            analysis = await self.analysis_agent.analyze_ticket(
                ticket_content=f"Subject: {ticket.subject}\n\nContent: {ticket.content}",
                customer_info=ticket.customer_info
            )
            
            # 2. Update context with analysis results
            self.context.update({
                "ticket_id": ticket.id,
                "analysis_timestamp": datetime.now().isoformat(),
                "category": analysis.category.value,
                "priority": analysis.priority.value,
                "customer_info": ticket.customer_info
            })
            
            # 3. Load appropriate response templates
            templates = self._load_response_templates()
            
            # 4. Generate response
            self.logger.info(f"Generating response for ticket {ticket.id}")
            response = await self.response_agent.generate_response(
                ticket_analysis=analysis,
                response_templates=templates,
                context=self.context
            )
            
            # 5. Calculate processing time
            processing_time = (datetime.now() - start_time).total_seconds()
            
            # 6. Create resolution object
            resolution = TicketResolution(
                ticket_id=ticket.id,
                analysis=analysis,
                response=response,
                processed_at=datetime.now(),
                processing_time=processing_time,
                status="completed"
            )
            
            # 7. Log completion
            self.logger.info(f"Successfully processed ticket {ticket.id} in {processing_time:.2f} seconds")
            
            return resolution
            
        except Exception as e:
            # Handle errors and create error resolution
            self.logger.error(f"Error processing ticket {ticket.id}: {str(e)}")
            return TicketResolution(
                ticket_id=ticket.id,
                analysis=None,
                response=None,
                processed_at=datetime.now(),
                processing_time=(datetime.now() - start_time).total_seconds(),
                status="error",
                error=str(e)
            )

    def _load_response_templates(self) -> Dict[str, str]:
        """Load response templates based on ticket category"""
        return {
            "access_issue": """
            Hello {name},
            
            I understand you're having trouble accessing the {feature}. Let me help you resolve this.
            
            {diagnosis}
            
            {resolution_steps}
            
            Priority Status: {priority_level}
            Estimated Resolution: {eta}
            
            Please let me know if you need any clarification.
            
            Best regards,
            Support Team
            """,
            
            "billing_inquiry": """
            Hi {name},
            
            Thank you for your inquiry about {billing_topic}.
            
            {explanation}
            
            {next_steps}
            
            If you have any questions, don't hesitate to ask.
            
            Best regards,
            Billing Team
            """,
            
            "feature_request": """
            Hello {name},
            
            Thank you for your feature suggestion regarding {feature_name}.
            
            {acknowledgment}
            
            {status_update}
            
            {timeline}
            
            We appreciate your input in making our product better.
            
            Best regards,
            Product Team
            """,
            
            "technical_issue": """
            Hi {name},
            
            Thank you for reporting the technical issue you're experiencing with {affected_component}.
            
            {technical_analysis}
            
            {solution_steps}
            
            Current Status: {status}
            Expected Resolution: {timeline}
            
            If you need immediate assistance, you can reach our technical team at:
            {support_contact}
            
            Best regards,
            Technical Support Team
            """,
            
            "urgent_issue": """
            URGENT RESPONSE
            
            Hello {name},
            
            We understand the critical nature of your issue regarding {issue_summary}.
            
            {immediate_actions}
            
            {escalation_status}
            
            We have assigned a dedicated specialist to your case:
            Specialist: {specialist_name}
            Direct Contact: {specialist_contact}
            
            We are treating this with highest priority and will provide updates every {update_frequency}.
            
            Urgent Support Line: {urgent_support_contact}
            
            Best regards,
            Senior Support Team
            """
        }

    async def batch_process_tickets(self, tickets: List[SupportTicket]) -> List[TicketResolution]:
        """Process multiple tickets concurrently"""
        self.logger.info(f"Starting batch processing of {len(tickets)} tickets")
        
        # Process tickets concurrently
        tasks = [self.process_ticket(ticket) for ticket in tickets]
        resolutions = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Log batch completion
        successful = sum(1 for r in resolutions if isinstance(r, TicketResolution) and r.status == "completed")
        self.logger.info(f"Batch processing completed. {successful}/{len(tickets)} tickets processed successfully")
        
        return resolutions

    def get_processing_stats(self) -> Dict[str, Any]:
        """Get statistics about processed tickets"""
        return {
            "total_processed": len(self.context),
            "last_processed": self.context.get("analysis_timestamp"),
            "context_size": len(json.dumps(self.context))
        }