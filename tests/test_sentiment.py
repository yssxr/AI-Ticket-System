import os
import json
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

# Load environment variables
load_dotenv()

def analyze_sentiment(text: str) -> float:
    client = InferenceClient(token=os.getenv('HF_API_KEY'))
    
    try:
        # Make API call for similarity computation
        response = client.post(
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
        
        # Parse the response - it comes as bytes containing a JSON string
        similarities = json.loads(response)
        print(f"Similarities: [positive: {similarities[0]:.4f}, negative: {similarities[1]:.4f}]")
        
        # Calculate sentiment (positive similarity minus negative similarity)
        sentiment_score = similarities[0] - similarities[1]
        
        return sentiment_score
            
    except Exception as e:
        print(f"Error in sentiment analysis: {str(e)}")
        return None

def main():
    # Test with your input
    text = input("Enter text to analyze: ")
    score = analyze_sentiment(text)
    print(f"\nSentiment Score: {score:.4f}")

if __name__ == '__main__':
    main()