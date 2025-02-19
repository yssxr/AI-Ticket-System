import asyncio
import json
import os
from datetime import datetime
from agents.ticket_processor import TicketProcessor, SupportTicket
from config import OPENAI_API_KEY, HF_API_KEY

def save_resolution_to_file(resolution, output_dir="output"):
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{output_dir}/ticket_{resolution.ticket_id}_{timestamp}.json"
    
    # Convert resolution to dictionary
    resolution_dict = {
        "ticket_id": resolution.ticket_id,
        "status": resolution.status,
        "processed_at": resolution.processed_at.isoformat(),
        "processing_time": resolution.processing_time
    }
    
    if resolution.status == "completed":
        resolution_dict.update({
            "analysis": {
                "category": resolution.analysis.category.value,
                "priority": resolution.analysis.priority.value,
                "key_points": resolution.analysis.key_points,
                "required_expertise": resolution.analysis.required_expertise,
                "sentiment": resolution.analysis.sentiment,
                "urgency_indicators": resolution.analysis.urgency_indicators,
                "business_impact": resolution.analysis.business_impact,
                "suggested_response_type": resolution.analysis.suggested_response_type
            },
            "response": {
                "text": resolution.response.response_text,
                "confidence_score": resolution.response.confidence_score,
                "requires_approval": resolution.response.requires_approval,
                "suggested_actions": resolution.response.suggested_actions
            }
        })
    else:
        resolution_dict["error"] = resolution.error
    
    # Save to file
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(resolution_dict, f, indent=2, ensure_ascii=False)
    
    return filename

async def main():
    # Create sample tickets
    sample_tickets = [
        SupportTicket(
            id="TKT-001",
            subject="Cannot access admin dashboard",
            content="""
            Hi Support,
            Since this morning I can't access the admin dashboard. I keep getting a 403 error.
            I need this fixed ASAP as I need to process payroll today.
            
            Thanks,
            John Smith
            Finance Director
            """,
            customer_info={
                "role": "Finance Director",
                "plan": "Enterprise",
                "company_size": "250+"
            }
        ),
        SupportTicket(
            id="TKT-002",
            subject="Question about billing cycle",
            content="""
            Hello,
            Our invoice shows billing from the 15th but we signed up on the 20th.
            Can you explain how the pro-rating works?
            
            Best regards,
            Sarah Jones
            """,
            customer_info={
                "role": "Billing Admin",
                "plan": "Professional",
                "company_size": "50-249"
            }
        )
    ]
    
    # Initialize the processor
    processor = TicketProcessor(OPENAI_API_KEY, HF_API_KEY)
    
    print("\nProcessing tickets individually:")
    print("-" * 50)
    
    for ticket in sample_tickets:
        print(f"\nProcessing ticket {ticket.id}...")
        resolution = await processor.process_ticket(ticket)
        
        # Save resolution to file
        output_file = save_resolution_to_file(resolution)
        
        print(f"\nResults for ticket {ticket.id}:")
        print(f"Status: {resolution.status}")
        if resolution.status == "completed":
            print(f"Category: {resolution.analysis.category.value}")
            print(f"Priority: {resolution.analysis.priority.value}")
            print(f"Sentiment Score: {resolution.analysis.sentiment:.2f}")
            print("\nKey Points:")
            for point in resolution.analysis.key_points:
                print(f"- {point}")
            
            print("\nGenerated Response:")
            print(resolution.response.response_text)
            
            print("\nSuggested Actions:")
            for action in resolution.response.suggested_actions:
                print(f"- {action}")
        else:
            print(f"Error: {resolution.error}")
        
        print(f"\nDetailed results saved to: {output_file}")
        print("-" * 50)
    
    # Get processing stats
    stats = processor.get_processing_stats()
    stats_file = "output/processing_stats.json"
    os.makedirs("output", exist_ok=True)
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2)
    
    print("\nProcessing Statistics:")
    print(json.dumps(stats, indent=2))
    print(f"\nStatistics saved to: {stats_file}")

if __name__ == "__main__":
    asyncio.run(main())