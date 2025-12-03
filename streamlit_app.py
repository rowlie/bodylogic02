# streamlit_app.py
import streamlit as st
import os
from rag_agent_logic import initialize_vectorstore, initialize_chain, chat_with_rag_and_tools

# ----------------------------
# PAGE CONFIG
# ----------------------------
st.set_page_config(
    page_title="RAG Agent with Tools",
    page_icon="🤖",
    layout="wide",
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chain_initialized" not in st.session_state:
    st.session_state.chain_initialized = False

if "chain" not in st.session_state:
    st.session_state.chain = None

# ----------------------------
# SIDEBAR
# ----------------------------
with st.sidebar:
    st.title("⚙️ Configuration")

    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    PINECONE_ENV = os.getenv("PINECONE_ENV")  # e.g., "us-west1-gcp"

    if not all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
        st.warning("⚠️ Missing API keys or Pinecone environment!")
    else:
        st.success("✅ All API keys are set")

    if st.button("🔄 Clear Chat History"):
        st.session_state.messages = []
        st.session_state.chain_initialized = False
        st.session_state.chain = None
        st.rerun()

# ----------------------------
# MAIN APP
# ----------------------------
st.title("🤖 Body Logic RAG Agent")
st.markdown("Ask me about your fitness goals or anything stored in the Pinecone database!")

# Initialize chain
if not st.session_state.chain_initialized:
    if all([OPENAI_API_KEY, PINECONE_API_KEY, PINECONE_ENV]):
        with st.spinner("Initializing RAG Agent..."):
            vectorstore = initialize_vectorstore(PINECONE_API_KEY, PINECONE_ENV)
            chain = initialize_chain(vectorstore)
            st.session_state.chain = chain
            st.session_state.chain_initialized = True
            st.success("Agent initialized successfully!")

# Display conversation
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("Ask your question here..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        try:
            answer, sources = chat_with_rag_and_tools(st.session_state.chain, prompt)
            message_placeholder.markdown(answer)

            st.session_state.messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            message_placeholder.error(f"Error: {str(e)}")
            st.session_state.messages.append({"role": "assistant", "content": f"Error: {str(e)}"})

# Footer
st.divider()
st.caption("💡 Tip: Memory and RAG retrieval are active!")
