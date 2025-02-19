import streamlit as st
import asyncio
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from agents.ticket_processor import TicketProcessor, SupportTicket

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Support Ticket Analyzer",
    page_icon="🎫",
    layout="wide"
)

def create_sentiment_chart(sentiment_score):
    # Create a range of points for x-axis
    x = np.linspace(-1, 1, 100)
    
    # Create a gaussian curve centered at the sentiment score
    y = np.exp(-((x - sentiment_score) ** 2) / 0.02)
    
    # Create DataFrame for plotting
    df = pd.DataFrame({
        'Value': x,
        'Density': y
    })
    
    # Create the chart
    chart = st.line_chart(
        df,
        x='Value',
        y='Density',
        height=100
    )
    
    # Add labels
    col1, col2, col3 = st.columns(3)
    col1.markdown("<div style='text-align: left'>Negative (-1)</div>", unsafe_allow_html=True)
    col2.markdown("<div style='text-align: center'>Neutral (0)</div>", unsafe_allow_html=True)
    col3.markdown("<div style='text-align: right'>Positive (1)</div>", unsafe_allow_html=True)

# Initialize processor
@st.cache_resource
def get_processor():
    return TicketProcessor(
        openai_api_key=os.getenv('OPENAI_API_KEY'),
        hf_api_key=os.getenv('HF_API_KEY')
    )

# Create the processor instance
processor = get_processor()

# Title
st.title("Support Ticket Analysis System")

# Create form for ticket input
with st.form("ticket_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        ticket_id = st.text_input("Ticket ID", value="TKT-001")
        subject = st.text_input("Subject", placeholder="Enter ticket subject")
        content = st.text_area("Content", placeholder="Enter ticket content", height=200)
    
    with col2:
        customer_role = st.selectbox(
            "Customer Role",
            ["User", "Admin", "Billing Admin", "Finance Director", "CEO"]
        )
        plan = st.selectbox(
            "Plan",
            ["Basic", "Professional", "Enterprise"]
        )
        company_size = st.selectbox(
            "Company Size",
            ["1-49", "50-249", "250+"]
        )
    
    submit = st.form_submit_button("Analyze Ticket")

# Process ticket when submitted
if submit:
    if not subject or not content:
        st.error("Please fill in both subject and content fields.")
    else:
        # Create ticket object
        ticket = SupportTicket(
            id=ticket_id,
            subject=subject,
            content=content,
            customer_info={
                "role": customer_role,
                "plan": plan,
                "company_size": company_size
            }
        )
        
        # Show processing message
        with st.spinner("Processing ticket..."):
            # Process ticket
            resolution = asyncio.run(processor.process_ticket(ticket))
        
        # Show results in tabs
        tab1, tab2, tab3 = st.tabs(["Analysis", "Response", "Debug Info"])
        
        with tab1:
            st.subheader("Ticket Analysis")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Category", resolution.analysis.category.value.title())
                st.metric("Priority", resolution.analysis.priority.value)
                st.metric("Processing Time", f"{resolution.processing_time:.2f}s")
            
            with col2:
                st.subheader("Sentiment Analysis")
                st.metric(
                    "Sentiment Score", 
                    f"{resolution.analysis.sentiment:.2f}",
                    delta=resolution.analysis.sentiment,
                    delta_color="normal"
                )
                
                #sentiment visualization
                create_sentiment_chart(resolution.analysis.sentiment)
            
            st.write("Key Points:")
            for point in resolution.analysis.key_points:
                st.markdown(f"- {point}")
            
            st.write("Business Impact:")
            st.info(resolution.analysis.business_impact)
            
            st.write("Required Expertise:")
            st.write(", ".join(resolution.analysis.required_expertise))
            
        with tab2:
            st.subheader("Generated Response")
            st.write("Confidence Score:", resolution.response.confidence_score)
            st.write("Requires Approval:", "✅" if resolution.response.requires_approval else "❌")
            st.text_area("Response", resolution.response.response_text, height=300)
            
            st.write("Suggested Actions:")
            for action in resolution.response.suggested_actions:
                st.markdown(f"- {action}")
        
        with tab3:
            st.subheader("Debug Information")
            st.json({
                "ticket_id": resolution.ticket_id,
                "status": resolution.status,
                "processed_at": resolution.processed_at.isoformat(),
                "urgency_indicators": resolution.analysis.urgency_indicators,
                "suggested_response_type": resolution.analysis.suggested_response_type
            })