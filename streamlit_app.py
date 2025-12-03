# -*- coding: utf-8 -*-
"""Streamlit App Interface (streamlit_app.py)"""

import streamlit as st
import os
from datetime import datetime
from typing import Dict, List

# Import your chain logic from the separate file
from rag_agent_logic import initialize_chain, chat_with_rag_and_tools

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state for conversation history display
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

# ============================================================================
# SIDEBAR
# ============================================================================

with st.sidebar:
    st.title("⚙️ Configuration")

    # Check if required API keys are set (for user clarity)
    api_key = os.getenv("OPENAI_API_KEY")
    pinecone_key = os.getenv("PINECONE_API_KEY")
    # LangSmith key is optional but recommended
    langsmith_key = os.getenv("LANGCHAIN_API_KEY") 

    if not all([api_key, pinecone_key]):
        st.warning("⚠️ Missing critical environment variables!")
        st.info(
            """
            Please ensure you have set these in your Streamlit Cloud secrets:
            - `OPENAI_API_KEY`
            - `PINECONE_API_KEY`
            - `LANGCHAIN_API_KEY` (Optional, for LangSmith tracing)
            """
        )
    else:
        st.success("✅ All critical API keys configured")

    st.divider()

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        
        # Manually clear the LangChain memory object in session state
        if "agent_memory" in st.session_state:
            del st.session_state["agent_memory"]
            
        st.session_state.chain_initialized = False
        st.rerun()

    st.divider()
    st.caption(
    """
**Demo Prompts**

**Pinecone Database:** *What is the most dangerous type of fat?*

**Tools (Automated):** *I am a 80kg male, light activity, and I want to lose weight. What should my calories and protein be?*

**Memory (Conversational):**
1. *I have 75 kg weight.*
2. *What was my weight just now?*
    """
)

# ============================================================================
# MAIN APPLICATION
# ============================================================================

st.title("🤖 Welcome to Body Logic ")
st.markdown(
    "Ask questions about your Fitness Goals - I can use tools and conversational memory and retrieve relevant content from hand curated videos."
)

# Initialize chain once
if not st.session_state.chain_initialized:
    try:
        with st.spinner("🔧 Initializing RAG Agent..."):
            initialize_chain()
            st.session_state.chain_initialized = True
            st.success("Agent successfully initialized! Ask me a question.")
    except Exception as e:
        st.error(f"❌ Failed to initialize chain. Check environment variables: {str(e)}")
        # Note: We stop execution here if the chain fails to prevent further errors.
        st.session_state.chain_initialized = False
        st.stop()

# Display conversation history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    # Add user message to history
    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt,
        }
    )

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()

        try:
            with st.spinner("Thinking..."):
                # Call your RAG chain
                response = chat_with_rag_and_tools(prompt)

                # Display response
                message_placeholder.markdown(response)

                # Add to history
                st.session_state.messages.append(
                    {
                        "role": "assistant",
                        "content": response,
                    }
                )
        except Exception as e:
            error_msg = f"❌ Error during response generation. Details: {str(e)}"
            message_placeholder.error(error_msg)
            st.session_state.messages.append(
                {
                    "role": "assistant",
                    "content": error_msg,
                }
            )

# Footer
st.divider()
st.caption("💡 Tip: Memory and tools are active. Try a follow-up question!")