import streamlit as st
import pandas as pd
import traceback
import os

from main import get_embedding_model, build_or_load_vectorstore, create_retriever, build_qa_chain, validate_api_key, JSON_DATA_DIR, GEMINI_MODEL, GEMINI_TEMPERATURE, GEMINI_MAX_TOKENS

def build_advanced_rag_chain(k=5):
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    
    api_key = validate_api_key()
    embeddings = get_embedding_model()
    # Provide the exact directory for JSON data
    vectorstore = build_or_load_vectorstore(embeddings, JSON_DATA_DIR)
    
    # Re-create retriever with dynamic k
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": k,
            "fetch_k": 20,
            "lambda_mult": 0.7,
        },
    )
    
    llm = ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=GEMINI_TEMPERATURE,
        max_output_tokens=GEMINI_MAX_TOKENS,
        convert_system_message_to_human=True,
        max_retries=5,
    )
    
    qa_chain = build_qa_chain(retriever, llm)
    return qa_chain
from langchain_experimental.agents.agent_toolkits import create_pandas_dataframe_agent
from langchain_google_genai import ChatGoogleGenerativeAI

# 1. Page Configuration and Styling
st.set_page_config(
    page_title="Telecom RAG Interface",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for a clean, modern SaaS look
st.markdown("""
<style>
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
    }
    /* Hide Deploy button and hamburger menu */
    .stDeployButton {display: none;}
    #MainMenu {display: none;}
    header {display: none;}

    /* ── Global page background ── */
    html, body, [data-testid="stAppViewContainer"],
    [data-testid="stApp"], .main, .main > div {
        background-color: #F1F5F9 !important;
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] * {
        color: #0F172A !important;
    }

    /* ── Chat message containers: always light background ── */
    [data-testid="stChatMessage"] {
        border-radius: 10px !important;
        padding: 1rem 1.25rem !important;
        margin-bottom: 0.5rem !important;
        border: 1px solid #E2E8F0 !important;
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }

    /* User message: slight blue tint */
    [data-testid="stChatMessage"]:has([data-testid="chatAvatarIcon-user"]) {
        background-color: #EFF6FF !important;
        border-color: #BFDBFE !important;
    }

    /* Force all text inside chat messages to be dark */
    [data-testid="stChatMessage"],
    [data-testid="stChatMessage"] *,
    [data-testid="stChatMessage"] p,
    [data-testid="stChatMessage"] span,
    [data-testid="stChatMessage"] div,
    [data-testid="stChatMessage"] li,
    [data-testid="stChatMessage"] strong,
    [data-testid="stChatMessage"] em,
    [data-testid="stChatMessage"] code,
    [data-testid="stChatMessage"] pre {
        color: #0F172A !important;
        background-color: transparent !important;
    }

    /* ── Expander (Show Sources) button – always visible ── */
    [data-testid="stChatMessage"] [data-testid="stExpander"],
    [data-testid="stChatMessage"] details {
        background-color: #F8FAFC !important;
        border: 1px solid #CBD5E1 !important;
        border-radius: 8px !important;
    }

    /* Expander header / summary – the clickable "Show Sources" button */
    [data-testid="stChatMessage"] details summary,
    [data-testid="stChatMessage"] [data-testid="stExpander"] summary,
    [data-testid="stChatMessage"] .streamlit-expanderHeader,
    [data-testid="stChatMessage"] [data-testid="stExpanderToggleIcon"],
    [data-testid="stChatMessage"] details summary *,
    [data-testid="stChatMessage"] .streamlit-expanderHeader * {
        color: #1E40AF !important;
        background-color: #EFF6FF !important;
        border-radius: 6px !important;
        font-weight: 600 !important;
    }

    /* Expander body text */
    [data-testid="stChatMessage"] details > div,
    [data-testid="stChatMessage"] .streamlit-expanderContent,
    [data-testid="stChatMessage"] .streamlit-expanderContent * {
        color: #0F172A !important;
        background-color: #F8FAFC !important;
    }

    /* General expanders outside chat (e.g., tab area) */
    [data-testid="stExpander"] {
        background-color: #FFFFFF !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 8px !important;
    }
    [data-testid="stExpander"] summary,
    .streamlit-expanderHeader {
        color: #1E40AF !important;
        font-weight: 600 !important;
        background-color: #EFF6FF !important;
    }
    [data-testid="stExpander"] summary *,
    .streamlit-expanderHeader * {
        color: #1E40AF !important;
    }
    [data-testid="stExpanderDetails"],
    .streamlit-expanderContent {
        background-color: #FFFFFF !important;
        color: #0F172A !important;
    }
    [data-testid="stExpanderDetails"] *,
    .streamlit-expanderContent * {
        color: #0F172A !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] button {
        color: #475569 !important;
        font-weight: 500 !important;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #1E40AF !important;
        border-bottom-color: #1E40AF !important;
    }

    /* ── Markdown & general text ── */
    p, li, span, label, div {
        color: #0F172A;
    }

    /* ── Code blocks ── */
    pre, code {
        background-color: #F1F5F9 !important;
        color: #0F172A !important;
        border-radius: 6px;
    }

    /* ── Spinner text ── */
    [data-testid="stSpinner"] p {
        color: #475569 !important;
    }

    /* ── Sidebar buttons (Reset Conversation Context) ── */
    [data-testid="stSidebar"] .stButton > button {
        background-color: #DC2626 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
        padding: 0.55rem 1rem !important;
        transition: background-color 0.2s ease, box-shadow 0.2s ease !important;
    }
    [data-testid="stSidebar"] .stButton > button:hover {
        background-color: #B91C1C !important;
        box-shadow: 0 4px 12px rgba(220, 38, 38, 0.35) !important;
        color: #FFFFFF !important;
    }
    [data-testid="stSidebar"] .stButton > button p,
    [data-testid="stSidebar"] .stButton > button span,
    [data-testid="stSidebar"] .stButton > button div {
        color: #FFFFFF !important;
    }
</style>
""", unsafe_allow_html=True)

# 2. State Management
if "chat_history_rag" not in st.session_state:
    st.session_state.chat_history_rag = []
    
if "chat_history_rca" not in st.session_state:
    st.session_state.chat_history_rca = []

def reset_conversation():
    st.session_state.chat_history_rag = []
    st.session_state.chat_history_rca = []
    st.rerun()

# 3. Cache Resource for the Retrieval Chain
@st.cache_resource(show_spinner=False)
def load_rag_chain(k_chunks):
    return build_advanced_rag_chain(k=k_chunks)

# 4. Sidebar Configuration
with st.sidebar:
    st.title("System Configuration")
    st.markdown("---")
    
    st.subheader("Retrieval Settings")
    retrieval_chunks = st.slider("Retrieval Chunk Count", min_value=1, max_value=15, value=5, step=1)
    
    st.markdown("---")
    st.subheader("System Status")
    
    # Initialize the chain and check status
    try:
        chain = load_rag_chain(k_chunks=retrieval_chunks)
        st.success("Database Connection: Active")
        st.success("Model Status: Loaded")
    except Exception as e:
        st.error("Database Connection: Disconnected")
        st.error(f"Error Details: {str(e)}")
        chain = None
        
    st.markdown("---")
    if st.button("Reset Conversation Context", use_container_width=True):
        reset_conversation()

# 5. Main Area and Tabs
st.title("Telecom Architecture Analysis")
tab1, tab2 = st.tabs(["3GPP Standards Expert", "O-RAN Root Cause Analysis"])

# --- TAB 1: 3GPP Standards Expert ---
with tab1:
    st.header("3GPP Technical Specifications")
    st.markdown("Query the vectorized 3GPP standards database.")
    
    # Render existing chat history
    for message in st.session_state.chat_history_rag:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "source_documents" in message:
                with st.expander("View Source Documents"):
                    for i, doc in enumerate(message["source_documents"]):
                        metadata = doc.metadata
                        spec_num = metadata.get("spec_number", "Unknown Spec")
                        release = metadata.get("release", "Unknown Release")
                        filename = metadata.get("filename", "Unknown File")
                        text_preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                        
                        st.markdown(f"**Document {i+1}: TS {spec_num} (Release {release})**")
                        st.markdown(f"*Filename: {filename}*")
                        st.text(text_preview)
                        st.markdown("---")

    # Chat Input
    if prompt := st.chat_input("Ask a question about 3GPP specifications...", key="rag_input"):
        # Append and display user message
        st.session_state.chat_history_rag.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
            
        # Assistant response
        with st.chat_message("assistant"):
            if chain is None:
                st.error("The retrieval chain failed to load. Please check your vector database and API keys.")
            else:
                with st.spinner("Querying vector database and reranking chunks..."):
                    try:
                        # Invoke the chain
                        response = chain.invoke({"query": prompt})
                        answer = response.get("result", "No answer generated.")
                        source_docs = response.get("source_documents", [])
                        
                        st.markdown(answer)
                        
                        # Display Source Documents Expander
                        if source_docs:
                            with st.expander("View Source Documents"):
                                for i, doc in enumerate(source_docs):
                                    metadata = doc.metadata
                                    spec_num = metadata.get("spec_number", "Unknown Spec")
                                    release = metadata.get("release", "Unknown Release")
                                    filename = metadata.get("filename", "Unknown File")
                                    text_preview = doc.page_content[:400] + "..." if len(doc.page_content) > 400 else doc.page_content
                                    
                                    st.markdown(f"**Document {i+1}: TS {spec_num} (Release {release})**")
                                    st.markdown(f"*Filename: {filename}*")
                                    st.text(text_preview)
                                    st.markdown("---")
                                    
                        # Save to session state
                        st.session_state.chat_history_rag.append({
                            "role": "assistant", 
                            "content": answer,
                            "source_documents": source_docs
                        })
                        
                    except Exception as e:
                        error_msg = str(e)
                        st.error("An error occurred while processing your request.")
                        if "429" in error_msg or "Quota" in error_msg:
                            st.warning("Actionable Step: API Rate Limit Exceeded. Please wait a minute before querying again, or check your billing plan.")
                        else:
                            st.warning("Actionable Step: Verify that your ChromaDB exists and your Google API Key is valid.")
                        st.code(traceback.format_exc())

# --- TAB 2: O-RAN Root Cause Analysis ---
with tab2:
    st.header("Telemetry Logs Analysis")
    st.markdown("Upload O-RAN telemetry CSV logs to perform Root Cause Analysis.")
    
    uploaded_file = st.file_uploader("Upload O-RAN Telemetry Logs", type=["csv"], key="csv_uploader")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.markdown("### Data Preview (First 5 Rows)")
            st.dataframe(df.head(5), use_container_width=True)
            
            # Initialize Pandas Agent
            llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
            agent = create_pandas_dataframe_agent(
                llm, 
                df, 
                verbose=False, 
                allow_dangerous_code=True
            )
            
            st.markdown("---")
            st.markdown("### Agent Interaction")
            
            # Render existing chat history
            for message in st.session_state.chat_history_rca:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            
            # RCA Chat Input
            if prompt_rca := st.chat_input("Ask about anomalies, latency spikes, or errors in the logs...", key="rca_input"):
                st.session_state.chat_history_rca.append({"role": "user", "content": prompt_rca})
                with st.chat_message("user"):
                    st.markdown(prompt_rca)
                    
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing telemetry data..."):
                        try:
                            result = agent.invoke({"input": prompt_rca})
                            answer_rca = result.get("output", "No analysis generated.")
                            st.markdown(answer_rca)
                            
                            st.session_state.chat_history_rca.append({"role": "assistant", "content": answer_rca})
                            
                        except Exception as e:
                            error_msg = str(e)
                            st.error("An error occurred during log analysis.")
                            if "429" in error_msg or "Quota" in error_msg:
                                st.warning("Actionable Step: API Rate Limit Exceeded. Please wait a minute or check your billing plan.")
                            else:
                                st.warning("Actionable Step: Check if the CSV structure matches the query parameters.")
                            st.code(traceback.format_exc())
                            
        except Exception as e:
            st.error("Failed to read the uploaded CSV file.")
            st.code(traceback.format_exc())
    else:
        st.info("Please upload a CSV file to begin analysis.")
