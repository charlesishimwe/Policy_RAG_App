import streamlit as st
import os
import requests
from dotenv import load_dotenv
# Load env vars for local testing
load_dotenv()
# Configuration
API_URL = os.getenv("API_URL", "http://localhost:8000") # Use Render URL when deployed
USE_LOCAL_PIPELINE = True # Set to False if you want to strictly test the FastAPI endpoint
st.set_page_config(
    page_title="HR Policy Assistant",
    page_icon="🤖",
    layout="centered"
)
st.title("🤖 HR Policy Assistant")
st.markdown("Ask questions about company policies. Answers are grounded in official documents with citations.")
# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message and message["sources"]:
            st.caption(f"📚 Sources: {', '.join(message['sources'])}")
# User input
if prompt := st.chat_input("Ask a question about company policies..."):
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    # Get AI response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                if USE_LOCAL_PIPELINE:
                    # Option A: Direct import (Faster for local demo, avoids CORS issues)
                    from app.pipeline import ask_question
                    
                    # Ensure API key is set for the pipeline
                    if not os.getenv("GROQ_API_KEY"):
                        st.error("⚠️ GROQ_API_KEY not found in environment variables.")
                        st.stop()
                        
                    result = ask_question(prompt)
                    answer = result["answer"]
                    sources = result["sources"]
                    
                else:
                    # Option B: Call FastAPI Endpoint (Better for testing deployment)
                    response = requests.post(
                        f"{API_URL}/chat",
                        json={"question": prompt},
                        timeout=30
                    )
                    response.raise_for_status()
                    data = response.json()
                    answer = data["answer"]
                    sources = data["sources"]
                # Display Answer
                st.write(answer)
                
                # Display Sources
                if sources:
                    st.caption(f"📚 Sources: {', '.join(sources)}")
                
                # Save assistant message
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"Sorry, I encountered an error: {str(e)}"
                })
