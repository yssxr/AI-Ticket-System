# AI Support Ticket Processing System

An AI-powered system that analyzes, prioritizes, and generates responses for customer support tickets using OpenAI's GPT-4 and Hugging Face's sentiment analysis.

## Setup Instructions

### Prerequisites
- Python 3.8+
- OpenAI API key
- Hugging Face API key

### Installation

1. Clone the repository:
```bash
git clone <https://github.com/yssxr/AI-Ticket-System>
cd AI-Ticket-System
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create configuration files:

Create a `.env` file in the root directory:
```env
OPENAI_API_KEY=your_openai_api_key_here
HF_API_KEY=your_huggingface_api_key_here
```


## Design Decisions

### Architecture

1. **Agent-Based Design**
   - Separate agents for analysis and response generation
   - Clear separation of concerns
   - Easy to extend with new capabilities

2. **Data Types**
   - Enums for categories and priorities
   - Dataclasses for structured data representation
   - Type hints throughout for better code maintainability

3. **Asynchronous Processing**
   - Async/await pattern for API calls
   - Batch processing capability for multiple tickets
   - Concurrent processing where possible

### Key Components

1. **TicketAnalysisAgent**
   - Uses GPT-4 for ticket classification and priority assessment
   - Hugging Face for accurate sentiment analysis
   - Extracts key points and business impact

2. **ResponseAgent**
   - Template-based response generation
   - Context-aware responses
   - Confidence scoring and approval flags

3. **TicketProcessor**
   - Orchestrates the analysis and response workflow
   - Maintains context between operations
   - Handles error cases and logging

### AI Integration

1. **OpenAI GPT-4**
   - Used for understanding ticket content
   - Function calling for structured outputs
   - Complex decision making (priority, categorization)

2. **Hugging Face Sentiment Analysis**
   - Sentence similarity based approach
   - Compares against positive/negative anchors
   - Provides numerical sentiment scores (-1 to 1 scale)

## Testing Approach

### Testing Tools
- Python's unittest framework
- Async test support
- Sample tickets with known characteristics
- Sentiment validation suite

### Running Tests

1. Run all tests:
```bash
python -m unittest discover tests
```

2. Run specific test file:
```bash
python tests/test_ticket_system.py
```

3. Run sentiment analysis test:
```bash
python tests/test_sentiment.py
```

## Usage

### Command Line Interface
Run the main script to process sample tickets:
```bash
python main.py
```

### Streamlit Interface
Run the interactive web interface:
```bash
streamlit run app.py
```

The Streamlit interface provides:
- Interactive ticket submission form
- Real-time analysis results
- Visual sentiment scale (-1 to 1)
- Response preview
- Debug information

### API Usage Example
```python
from agents.ticket_processor import TicketProcessor, SupportTicket

# Initialize processor
processor = TicketProcessor(OPENAI_API_KEY, HF_API_KEY)

# Create a ticket
ticket = SupportTicket(
    id="TKT-001",
    subject="Cannot access dashboard",
    content="Getting 403 error...",
    customer_info={"role": "Admin"}
)

# Process ticket
resolution = await processor.process_ticket(ticket)

# Check results
print(f"Category: {resolution.analysis.category}")
print(f"Priority: {resolution.analysis.priority}")
print(f"Sentiment: {resolution.analysis.sentiment}")
print(f"Response: {resolution.response.response_text}")
```

### Batch Processing
```python
tickets = [ticket1, ticket2, ticket3]
resolutions = await processor.batch_process_tickets(tickets)
```

## Future Improvements

1. Potential Enhancements
   - Multi-language support
   - Response quality metrics

2. Performance Optimizations
   - Caching for similar tickets
   - Batch API calls
   - Response template optimization

3. Additional Features
   - Dashboard for metrics
   - Custom response templates
   - Integration with ticketing systems