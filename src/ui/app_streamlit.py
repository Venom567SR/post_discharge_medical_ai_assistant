import streamlit as st
import requests
import uuid
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Post-Discharge Medical AI Assistant",
    page_icon="🏥",
    layout="wide"
)

# Custom CSS for cleaner, Perplexity-inspired aesthetic
st.markdown("""
<style>
    /* Cleaner chat messages */
    .stChatMessage {
        padding: 1.5rem;
        border-radius: 0.5rem;
    }
    
    /* Citation badges */
    code {
        background-color: #f0f2f6;
        padding: 0.2rem 0.4rem;
        border-radius: 0.25rem;
        font-weight: 600;
    }
    
    /* Source cards */
    .stExpander {
        border: 1px solid #e6e9ef;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    
    /* Status messages */
    .stAlert {
        border-radius: 0.5rem;
        border-left: 4px solid #4f8bf9;
    }
    
    /* Dividers in source list */
    hr {
        margin: 0.5rem 0;
        border-color: #e6e9ef;
    }
    
    /* Clean typography */
    .stMarkdown {
        line-height: 1.6;
    }
</style>
""", unsafe_allow_html=True)

# API endpoint
API_URL = "http://localhost:8000"

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())

if "messages" not in st.session_state:
    st.session_state.messages = []
    # Add initial greeting from receptionist
    st.session_state.messages.append({
        "role": "assistant",
        "content": "Hello! I'm your post-discharge care assistant. What's your name?",
        "metadata": {
            "agent": "receptionist",
            "handoffs": [],
            "sources": []
        }
    })

if "user_id" not in st.session_state:
    st.session_state.user_id = f"user_{uuid.uuid4().hex[:8]}"


def check_api_health():
    """Check if API is available."""
    try:
        response = requests.get(f"{API_URL}/health", timeout=5)
        return response.status_code == 200, response.json()
    except Exception as e:
        return False, {"error": str(e)}


def send_message(message: str, rag_enabled: bool, web_search_enabled: bool):
    """Send message to API and get response."""
    try:
        response = requests.post(
            f"{API_URL}/chat",
            json={
                "user_id": st.session_state.user_id,
                "message": message,
                "session_id": st.session_state.session_id,
                "rag_enabled": rag_enabled,
                "web_search_enabled": web_search_enabled
            },
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            return {
                "answer": f"Error: API returned status {response.status_code}",
                "sources": [],
                "agent": "error",
                "handoffs": []
            }
    except Exception as e:
        return {
            "answer": f"Error connecting to API: {str(e)}",
            "sources": [],
            "agent": "error",
            "handoffs": []
        }


# Sidebar
with st.sidebar:
    st.title("🏥 Medical AI Assistant")
    st.markdown("---")
    
    # API Health Check
    st.subheader("System Status")
    is_healthy, health_data = check_api_health()
    
    if is_healthy:
        st.success("✅ API Connected")
        if "llm_available" in health_data:
            if health_data["llm_available"]:
                st.info("🤖 LLM Available")
            else:
                st.warning("⚠️ LLM in Stub Mode")
        
        if "web_search_available" in health_data:
            if health_data["web_search_available"]:
                st.info("🔍 Web Search Available")
            else:
                st.warning("⚠️ Web Search in Stub Mode")
    else:
        st.error("❌ API Not Connected")
        st.caption("Make sure the API is running: `uvicorn src.api.main:app --reload`")
    
    st.markdown("---")
    
    # Settings
    st.subheader("Settings")
    
    rag_enabled = st.checkbox("Use RAG (Reference Database)", value=True)
    web_search_enabled = st.checkbox("Use Web Search", value=True)
    
    st.markdown("---")
    
    # Session Info
    st.subheader("Session Info")
    st.caption(f"User: {st.session_state.user_id}")
    st.caption(f"Session: {st.session_state.session_id[:8]}...")
    
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.session_state.session_id = str(uuid.uuid4())
        st.rerun()
    
    st.markdown("---")
    
    # Help
    with st.expander("💡 How to Use"):
        st.markdown("""
        **Getting Started:**
        1. Introduce yourself: "Hi, my name is John Smith"
        2. Answer follow-up questions from the receptionist
        3. Ask medical questions to get evidence-based answers
        
        **Example Questions:**
        - "What are the early signs of kidney dysfunction?"
        - "How should I manage my medications?"
        - "What are the latest treatment guidelines for CKD?"
        
        **Features:**
        - 🤖 Multi-agent system (Receptionist + Clinical)
        - 📚 RAG over medical references
        - 🔍 Web search for current information
        - 📝 Inline citations
        """)


# Main content
st.title("Post-Discharge Medical AI Assistant")
st.caption("AI-powered support for post-discharge patient care")

# Display medical disclaimer at top
st.warning("⚠️ **Medical Disclaimer**: This assistant is for educational purposes only. Always consult healthcare professionals for medical advice.")

st.markdown("---")

# Chat container
chat_container = st.container()

with chat_container:
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            
            # Display metadata for assistant messages
            if message["role"] == "assistant" and "metadata" in message:
                metadata = message["metadata"]
                
                # Display agent info with emoji
                if "agent" in metadata:
                    agent_emoji = "👋" if metadata["agent"] == "receptionist" else "🏥"
                    st.caption(f"{agent_emoji} Agent: {metadata['agent'].title()}")
                
                # Display handoffs
                if metadata.get("handoffs"):
                    st.caption(f"🔄 Handoff: {' → '.join(metadata['handoffs'])}")
                
                # Display sources in Perplexity style
                if "sources" in metadata and metadata["sources"]:
                    with st.expander(f"📚 Sources ({len(metadata['sources'])})", expanded=False):
                        for i, source in enumerate(metadata["sources"], 1):
                            if source["source_type"] == "reference":
                                ref_id = source.get("reference_id", "Reference")
                                page = source.get("page")
                                score = source.get("score")
                                
                                # Create numbered citation badge
                                col1, col2 = st.columns([0.1, 0.9])
                                with col1:
                                    st.markdown(f"**`[{i}]`**")
                                with col2:
                                    source_text = f"**{ref_id}**"
                                    if page:
                                        source_text += f", page {page}"
                                    if score:
                                        # Show relevance as percentage
                                        relevance_pct = int(score * 100)
                                        source_text += f" • {relevance_pct}% relevant"
                                    st.markdown(source_text)
                                    
                                    if source.get("snippet"):
                                        st.caption(f"_{source['snippet']}_")
                                
                                if i < len(metadata["sources"]):
                                    st.divider()
                            
                            elif source["source_type"] == "web":
                                url = source.get("url", "N/A")
                                col1, col2 = st.columns([0.1, 0.9])
                                with col1:
                                    st.markdown(f"**`[{i}]`**")
                                with col2:
                                    st.markdown(f"🌐 **Web Source**")
                                    st.markdown(f"[{url}]({url})")
                                
                                if i < len(metadata["sources"]):
                                    st.divider()
                            
                            elif source["source_type"] == "web_stub":
                                col1, col2 = st.columns([0.1, 0.9])
                                with col1:
                                    st.markdown(f"**`[{i}]`**")
                                with col2:
                                    st.markdown("⚠️ Web search unavailable (API key not configured)")
                
                # Show processing steps for clinical responses (transparency)
                if metadata.get("processing_steps"):
                    steps = metadata["processing_steps"]
                    if steps:
                        with st.expander("🔍 How this answer was researched", expanded=False):
                            if "searching_references" in steps:
                                rag_count = metadata.get("rag_chunks", 0)
                                st.markdown(f"✅ Searched medical reference database ({rag_count} sources)")
                            if "searching_web" in steps:
                                web_count = metadata.get("web_results", 0)
                                st.markdown(f"✅ Searched web for recent information ({web_count} results)")
                            st.markdown(f"✅ Synthesized answer with citations")

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Check API health before sending
    is_healthy, _ = check_api_health()
    
    if not is_healthy:
        st.error("❌ Cannot send message: API is not connected. Please start the API server.")
    else:
        # Add user message to chat
        st.session_state.messages.append({
            "role": "user",
            "content": prompt
        })
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # Get response from API with status messages
        with st.chat_message("assistant"):
            # Create placeholders for status messages
            status_placeholder = st.empty()
            response_placeholder = st.empty()
            
            # Show initial analyzing status
            with status_placeholder:
                st.info("🤔 Analyzing your query...")
            
            # Send message to API
            response = send_message(prompt, rag_enabled, web_search_enabled)
            
            # Check if this is a handoff to clinical
            if response.get("agent") == "clinical" and "receptionist->clinical" in response.get("handoffs", []):
                with status_placeholder:
                    st.info("🏥 This sounds like a medical concern. Let me connect you with our Clinical AI Agent...")
                import time
                time.sleep(1.5)  # Brief pause to show handoff message
            
            # Show searching status for clinical queries
            if response.get("agent") == "clinical":
                if rag_enabled:
                    sources_count = response.get("metadata", {}).get("rag_chunks", 0)
                    if sources_count > 0:
                        with status_placeholder:
                            st.info(f"📚 Searching medical references... (considering {sources_count} sources)")
                        import time
                        time.sleep(0.8)
                
                if web_search_enabled and response.get("metadata", {}).get("web_results", 0) > 0:
                    with status_placeholder:
                        st.info("🔍 Looking for recent information...")
                    import time
                    time.sleep(0.8)
            
            # Clear status and show response
            status_placeholder.empty()
            
            # Display response
            with response_placeholder:
                st.markdown(response["answer"])
            
            # Display metadata
            agent = response.get("agent", "unknown")
            agent_emoji = "👋" if agent == "receptionist" else "🏥"
            st.caption(f"{agent_emoji} Agent: {agent.title()}")
            
            if response.get("handoffs"):
                st.caption(f"🔄 Handoff: {' → '.join(response['handoffs'])}")
            
            # Display sources in Perplexity style
            if response.get("sources"):
                with st.expander(f"📚 Sources ({len(response['sources'])})", expanded=False):
                    for i, source in enumerate(response["sources"], 1):
                        if source["source_type"] == "reference":
                            ref_id = source.get("reference_id", "Reference")
                            page = source.get("page")
                            score = source.get("score")
                            
                            # Numbered citation badge
                            col1, col2 = st.columns([0.1, 0.9])
                            with col1:
                                st.markdown(f"**`[{i}]`**")
                            with col2:
                                source_text = f"**{ref_id}**"
                                if page:
                                    source_text += f", page {page}"
                                if score:
                                    relevance_pct = int(score * 100)
                                    source_text += f" • {relevance_pct}% relevant"
                                st.markdown(source_text)
                                
                                if source.get("snippet"):
                                    st.caption(f"_{source['snippet']}_")
                            
                            if i < len(response["sources"]):
                                st.divider()
                        
                        elif source["source_type"] == "web":
                            url = source.get("url", "N/A")
                            col1, col2 = st.columns([0.1, 0.9])
                            with col1:
                                st.markdown(f"**`[{i}]`**")
                            with col2:
                                st.markdown(f"🌐 **Web Source**")
                                st.markdown(f"[{url}]({url})")
                            
                            if i < len(response["sources"]):
                                st.divider()
                        
                        elif source["source_type"] == "web_stub":
                            col1, col2 = st.columns([0.1, 0.9])
                            with col1:
                                st.markdown(f"**`[{i}]`**")
                            with col2:
                                st.markdown("⚠️ Web search unavailable (API key not configured)")
            
            # Show processing transparency for clinical responses
            if response.get("metadata", {}).get("processing_steps"):
                steps = response["metadata"]["processing_steps"]
                if steps:
                    with st.expander("🔍 How this answer was researched", expanded=False):
                        if "searching_references" in steps:
                            rag_count = response["metadata"].get("rag_chunks", 0)
                            st.markdown(f"✅ Searched medical reference database ({rag_count} sources)")
                        if "searching_web" in steps:
                            web_count = response["metadata"].get("web_results", 0)
                            st.markdown(f"✅ Searched web for recent information ({web_count} results)")
                        st.markdown(f"✅ Synthesized answer with citations")
        
        # Add assistant message to chat
        st.session_state.messages.append({
            "role": "assistant",
            "content": response["answer"],
            "metadata": {
                "agent": response.get("agent", "unknown"),
                "handoffs": response.get("handoffs", []),
                "sources": response.get("sources", [])
            }
        })
        
        # Rerun to update chat
        st.rerun()

# Footer
st.markdown("---")
st.caption("🏥 Post-Discharge Medical AI Assistant | Built with LangGraph, Qdrant, Gemini, and Streamlit")
st.caption("⚠️ This is a demonstration system. Do not rely on it for actual medical decisions.")