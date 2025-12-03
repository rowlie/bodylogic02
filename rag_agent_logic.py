# -*- coding: utf-8 -*-
"""
Stable RAG Agent Logic – LangChain 0.2.x compatible
Includes:
- LangChain AgentExecutor with tools
- ConversationBufferWindowMemory stored in Streamlit session_state
- Pinecone RAG retrieval
- Fully typed @tool functions
"""

import os
from datetime import datetime
from typing import Dict, List
import streamlit as st

# --- Embeddings / Vector DB ---
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone

# --- LangChain ---
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage
from langchain_core.tools import tool
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.memory import ConversationBufferWindowMemory

# ============================================================================
# CONFIGURATION
# ============================================================================

INDEX_NAME = "youtube-qa-index"
TOP_K = 5

SYSTEM_PROMPT = """
You are a friendly, evidence-based personal trainer and RAG assistant.

Goals:
1. Provide safe, practical fitness advice.
2. Tailor suggestions to the user's level and goals.
3. Explain reasoning in simple language.

Use RAG context when available:
- RAG context is prefixed with 'RAG_CONTEXT:'
- User input is prefixed with 'USER_QUERY:'

TOOLS:
- Use `calculator` for math.
- Use `word_count` for counting words.
- Use `convert_case` for text case changes.
- Use `get_current_time` for date/time.
- Use ONLY `estimate_targets` for calories/protein targets.

When a tool is used, mention it and base your answer on the tool output.
"""

# ============================================================================
# GLOBALS
# ============================================================================

_initialized = False
retriever: SentenceTransformer = None
pc: Pinecone = None
index = None
rag_agent_chain = None

# ============================================================================
# ENVIRONMENT SETUP
# ============================================================================

def _setup_env() -> None:
    os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")


@st.cache_resource
def get_retriever() -> SentenceTransformer:
    print("📥 Loading SentenceTransformer (CPU-only)...")
    model = SentenceTransformer("sentence-transformers/all-mpnet-base-v2", device="cpu")
    print("✅ Loaded all-mpnet-base-v2")
    return model

# ============================================================================
# TOOLS
# ============================================================================

@tool
def calculator(expression: str) -> str:
    """Evaluate a simple math expression."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {str(e)}"

@tool
def get_current_time() -> str:
    """Return the current date/time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@tool
def word_count(text: str) -> str:
    """Count words in text."""
    return f"Word count: {len(text.split())}"

@tool
def convert_case(text: str, case_type: str) -> str:
    """
    Convert text case.
    case_type options: 'upper', 'lower', or 'title'
    """
    if case_type == "upper":
        return text.upper()
    elif case_type == "lower":
        return text.lower()
    elif case_type == "title":
        return text.title()
    else:
        return f"Error: Unknown case_type '{case_type}'"

@tool
def estimate_targets(
    weight_kg: float, 
    sex: str, 
    activity: str, 
    goal: str
) -> str:
    """Estimate daily calories and protein targets."""
    factors = {"sedentary": 28, "light": 31, "moderate": 34, "active": 37}
    factor = factors.get(activity.lower(), 31)
    maintenance = weight_kg * factor

    if goal.lower() == "lose":
        calories = maintenance - 400
        goal_text = "weight loss"
    elif goal.lower() == "gain":
        calories = maintenance + 400
        goal_text = "muscle gain"
    else:
        calories = maintenance
        goal_text = "maintenance"

    protein_low = weight_kg * 1.6
    protein_high = weight_kg * 2.2

    return (
        f"Estimated daily targets for {goal_text}:\n"
        f"- Calories: {int(calories)} kcal/day\n"
        f"- Protein: {protein_low:.1f}–{protein_high:.1f} g/day"
    )

tools = [calculator, get_current_time, word_count, convert_case, estimate_targets]

# ============================================================================
# RAG HELPERS
# ============================================================================

def retrieve_pinecone_context(query: str, top_k: int = TOP_K) -> Dict:
    global index
    if index is None:
        return {"matches": []}

    try:
        vec = get_retriever().encode(query).tolist()
        res = index.query(vector=vec, top_k=top_k, include_metadata=True)
        return res
    except Exception as e:
        print("⚠️ Pinecone retrieval error:", e)
        return {"matches": []}

def context_string_from_matches(matches: List) -> str:
    ctx = []
    for m in matches:
        passage = m.get("metadata", {}).get("text") or ""
        if passage:
            ctx.append(passage)
    return "\n\n".join(ctx)

def _retrieve_and_format_context(user_message: str) -> dict:
    pc_res = retrieve_pinecone_context(user_message)
    ctx = context_string_from_matches(pc_res.get("matches", []))

    rag_block = f"RAG_CONTEXT:\n{ctx}\n\n" if ctx else ""
    final_input = f"{rag_block}USER_QUERY: {user_message}"

    return {"input": final_input, "rag_context": ctx}

# ============================================================================
# INITIALIZATION
# ============================================================================

def initialize_chain() -> None:
    global _initialized, pc, index, rag_agent_chain

    if _initialized:
        return

    _setup_env()
    print("🔧 Initializing RAG + Tools Agent...")

    # Pinecone setup
    pinecone_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_key:
        raise ValueError("PINECONE_API_KEY not set")
    pc = Pinecone(api_key=pinecone_key)
    index = pc.Index(INDEX_NAME)
    print(f"✅ Connected to Pinecone index: {INDEX_NAME}")

    # LLM setup
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )

    # Memory
    if "agent_memory" not in st.session_state:
        st.session_state.agent_memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            return_messages=True,
            k=20,
        )
    memory_obj = st.session_state.agent_memory

    # Prompt template
    prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(content=SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ]
    )

    # Agent executor
    agent = create_openai_functions_agent(llm, tools, prompt)
    executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory_obj,
        handle_parsing_errors=True,
    )

    # RAG → Agent chain
    rag_agent_chain = (
        RunnableLambda(_retrieve_and_format_context)
        | RunnablePassthrough.assign(final_response=executor)
    )

    _initialized = True

# ============================================================================
# MAIN CHAT FUNCTION
# ============================================================================

def chat_with_rag_and_tools(user_message: str) -> str:
    global rag_agent_chain
    if not _initialized or rag_agent_chain is None:
        raise RuntimeError("Chain not initialized. Call initialize_chain()")

    result = rag_agent_chain.invoke(user_message)
    return result["final_response"].get("output")
