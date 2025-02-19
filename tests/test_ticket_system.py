import unittest
import asyncio
import os
import sys
from datetime import datetime
from dotenv import load_dotenv


# Add the project root directory to Python path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, PROJECT_ROOT)

# Import our system components
from agents.ticket_processor import TicketProcessor, SupportTicket
from agents.data_types import TicketCategory, Priority

# Load environment variables
load_dotenv()

# Sample test data
SAMPLE_TICKETS = [
    {
        "id": "TKT-001",
        "subject": "Cannot access admin dashboard",
        "content": """
        Hi Support,
        Since this morning I can't access the admin dashboard. I keep getting a 403 error.
        I need this fixed ASAP as I need to process payroll today.
        
        Thanks,
        John Smith
        Finance Director
        """,
        "customer_info": {
            "role": "Admin",
            "plan": "Enterprise",
            "company_size": "250+"
        }
    },
    {
        "id": "TKT-002",
        "subject": "Question about billing cycle",
        "content": """
        Hello,
        Our invoice shows billing from the 15th but we signed up on the 20th.
        Can you explain how the pro-rating works?
        
        Best regards,
        Sarah Jones
        """,
        "customer_info": {
            "role": "Billing Admin",
            "plan": "Professional",
            "company_size": "50-249"
        }
    }
]

class TestTicketSystem(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up test resources that can be shared across all tests"""
        cls.openai_api_key = os.getenv('OPENAI_API_KEY')
        cls.hf_api_key = os.getenv('HF_API_KEY')
        
        if not cls.openai_api_key or not cls.hf_api_key:
            raise ValueError("API keys not found in environment variables")
            
        cls.processor = TicketProcessor(cls.openai_api_key, cls.hf_api_key)
        
    def setUp(self):
        """Set up test fixtures before each test method"""
        self.sample_tickets = [
            SupportTicket(**ticket) for ticket in SAMPLE_TICKETS
        ]
    
    def test_ticket_creation(self):
        """Test that tickets are created correctly"""
        ticket = self.sample_tickets[0]
        self.assertEqual(ticket.id, "TKT-001")
        self.assertEqual(ticket.customer_info["role"], "Admin")
        self.assertIn("403 error", ticket.content)

    
    
    
    async def async_test_ticket_analysis(self):
        """Test ticket analysis functionality"""
        ticket = self.sample_tickets[0]
        resolution = await self.processor.process_ticket(ticket)
        
        # Test basic resolution properties
        self.assertEqual(resolution.status, "completed")
        self.assertEqual(resolution.ticket_id, ticket.id)
        self.assertIsNotNone(resolution.processed_at)
        self.assertGreater(resolution.processing_time, 0)
        
        # Test analysis results
        analysis = resolution.analysis
        self.assertEqual(analysis.category, TicketCategory.ACCESS)
        self.assertEqual(analysis.priority, Priority.URGENT)  # Should be urgent due to payroll impact
        self.assertGreaterEqual(len(analysis.key_points), 1)
        self.assertIsInstance(analysis.sentiment, float)
        self.assertTrue(-1 <= analysis.sentiment <= 1)
        
        # Test response generation
        response = resolution.response
        self.assertIsNotNone(response.response_text)
        self.assertTrue(0 <= response.confidence_score <= 1)
        self.assertIsInstance(response.requires_approval, bool)
        self.assertGreaterEqual(len(response.suggested_actions), 1)
    
    async def async_test_billing_ticket(self):
        """Test handling of billing-related tickets"""
        ticket = self.sample_tickets[1]
        resolution = await self.processor.process_ticket(ticket)
        
        # Test categorization and priority
        self.assertEqual(resolution.analysis.category, TicketCategory.BILLING)
        self.assertLess(resolution.analysis.priority, Priority.URGENT)  # Billing should not be urgent
        
        # Test response content
        self.assertIn("pro-rat", resolution.response.response_text.lower())
        self.assertIn("billing", resolution.response.response_text.lower())
    
    async def async_test_batch_processing(self):
        """Test batch processing of multiple tickets"""
        resolutions = await self.processor.batch_process_tickets(self.sample_tickets)
        
        self.assertEqual(len(resolutions), len(self.sample_tickets))
        for resolution in resolutions:
            self.assertEqual(resolution.status, "completed")

    async def _analyze_sentiment(self, text: str) -> float:
        try:
            # Define reference texts for sentiment comparison
            positive_text = "I am very happy and satisfied with the service"
            negative_text = "I am very frustrated and unhappy with the service"
            
            # Make API call with correct parameters for computing similarity
            response = self.hf_client.post(
                json={
                    "inputs": {
                        "source_sentence": text,
                        "sentences": [positive_text, negative_text]
                    }
                },
                model="sentence-transformers/all-MiniLM-L6-v2",
                task="sentence-similarity"
            )
            
            # Response will be a list of similarity scores
            similarities = response
            
            # Calculate sentiment score between -1 and 1
            # First score is similarity to positive, second to negative
            sentiment_score = similarities[0] - similarities[1]
            
            # Ensure the score is between -1 and 1
            return max(min(sentiment_score, 1.0), -1.0)
            
        except Exception as e:
            print(f"Error in sentiment analysis: {str(e)}")
            print("Falling back to OpenAI for sentiment analysis...")
            
            try:
                # Fallback to OpenAI
                messages = [
                    {"role": "system", "content": "Analyze the sentiment of the following text and respond with a single number between -1 (very negative) and 1 (very positive), where 0 is neutral. Only respond with the number, no explanation."},
                    {"role": "user", "content": text}
                ]
                
                response = self.openai_client.chat.completions.create(
                    model="gpt-4-turbo-preview",
                    messages=messages,
                    max_tokens=10
                )
                
                sentiment = float(response.choices[0].message.content.strip())
                return max(min(sentiment, 1.0), -1.0)
                
            except Exception as e2:
                print(f"Error in OpenAI fallback sentiment analysis: {str(e2)}")
                return 0.0  # Final fallback to neutral sentiment
    
    def test_stats_tracking(self):
        """Test that processing statistics are tracked correctly"""
        stats = self.processor.get_processing_stats()
        
        self.assertIn("total_processed", stats)
        self.assertIn("last_processed", stats)
        self.assertIn("context_size", stats)
    
    def test_runner(self):
        """Run all async tests"""
        async def run_async_tests():
            await self.async_test_ticket_analysis()
            await self.async_test_billing_ticket()
            await self.async_test_batch_processing()
        
        asyncio.run(run_async_tests())

def main():
    unittest.main(verbosity=2)

if __name__ == '__main__':
    main()