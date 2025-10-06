# Consigli_2\main.py


########## IMPORTS ##########

# Standard Library Imports
import os
import sys
import logging
from pathlib import Path

# Third-party Imports
import streamlit as st
from dotenv import load_dotenv

# Local Application/Library-Specific Import
from src.rag_engine import RAGEngine
from src.vector_store import VectorStoreManager
from src.document_processor import DocumentProcessor


########## CONFIGs ##########

# Load environment variables
load_dotenv()

# Initialise logger
logger = logging.getLogger(__name__)

# Add src to path
sys.path.append(str(Path(__file__).parent))

# Page config
st.set_page_config(
    page_title="Automotive Financial Analyst",
    layout="wide",
    initial_sidebar_state="expanded")

# Custom CSS for better presentation
st.markdown("""
    <style>
        /* Header and subtitle */
        .header-container{
            text-align: center;
            padding: 1.25rem 0 0.5rem 0;
            margin-bottom: 0.75rem;
        }

        .main-header {
            font-size: 2.4rem;
            font-weight: 800;
            margin: 0.2rem 0 0.4rem 0;
            /* subtle gradient text for visual polish */
            background: linear-gradient(90deg, #ca2332 0%, #1f77b4 100%);
            -webkit-background-clip: text;
            background-clip: text;
            color: transparent;
            letter-spacing: -0.5px;
        }

        .subtitle {
            font-size: 1.05rem;
            font-weight: 800;
            color: #1f77b4;
            margin: 0;
            opacity: 0.95;
        }

        /* Info box */
        .info-box {
            background-color: #f7fbff;
            padding: 1.75rem;
            border-radius: 0.6rem;
            margin: 1rem 0;
            border-left: 5px solid #1f77b4;
            box-shadow: 0 6px 18px rgba(31,119,180,0.06);
        }
        .info-box h2 {
            color: #1f77b4;
            margin-top: 0;
        }
        .info-box p {
            color: #333333;
            font-size: 1.05rem;
            line-height: 1.45;
            margin: 0.35rem 0;
        }

        /* Slightly tighten spacing for the sidebar and main columns */
        .stSidebar {
            padding-top: 0.5rem;
        }

        /* Chat bubble improvements */
        .stChatMessage > div {
            line-height: 1.5 !important;
        }

        /* Make sample question buttons sit nicely */
        .sample-btn {
            width: 100%;
            text-align: left;
        }

        /* Responsive tweaks */
        @media (max-width: 600px) {
            .main-header { font-size: 1.5rem; }
            .subtitle { font-size: 0.95rem; }
        }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []
if 'rag_engine' not in st.session_state:
    st.session_state.rag_engine = None
if 'initialized' not in st.session_state:
    st.session_state.initialized = False
if 'model_provider' not in st.session_state:
    # Check which API key is available
    if os.getenv("GROQ_API_KEY"):
        st.session_state.model_provider = "groq"
    elif os.getenv("HF_API_KEY"):
        st.session_state.model_provider = "huggingface"
    else:
        st.session_state.model_provider = None


########## HELPER FUNCTIONS ##########

def initialize_rag_system():
    """Initialize the RAG system with documents"""

    # Check for API keys
    if st.session_state.model_provider is None:
        st.error("No API key found! Please add GROQ_API_KEY or HF_API_KEY to your .env file")
        st.stop()

    with st.spinner("Initializing RAG system. Please wait!"):
        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            # Initialize components
            status_text.text("Loading document processor.")
            progress_bar.progress(20)
            doc_processor = DocumentProcessor()

            status_text.text("Initializing vector store.")
            progress_bar.progress(40)
            vector_store = VectorStoreManager()

            # Process documents
            status_text.text("Processing PDF documents.")
            progress_bar.progress(50)
            data_path = Path("data")

            if not data_path.exists():
                st.error(f"Data folder not found at: {data_path.absolute()}")
                st.stop()

            documents = doc_processor.load_and_process_documents(data_path)

            if not documents:
                st.error("No documents were processed. Please check your PDF files.")
                st.stop()

            status_text.text(f"Processed {len(documents)} document chunks.")
            progress_bar.progress(70)

            # Create vector store
            status_text.text("Creating vector embeddings and index.")
            progress_bar.progress(70)
            vector_store.create_index(documents)

            # Show GPU info if available
            device_info = vector_store.get_device_info()
            if device_info['device'] == 'cuda':
                st.info(f"GPU Acceleration: {device_info.get('gpu_name', 'NVIDIA GPU')}.")

            progress_bar.progress(90)

            # Initialize RAG engine
            status_text.text(f"Initializing LLM ({st.session_state.model_provider}).")
            rag_engine = RAGEngine(
                vector_store,
                model_provider=st.session_state.model_provider
            )

            progress_bar.progress(100)
            status_text.text("System ready!")

            st.session_state.rag_engine = rag_engine
            st.session_state.initialized = True

            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()

            st.success(f"""
                         **System Initialized Successfully!**
                        - Documents processed: {len(documents)} chunks
                        - Companies: BMW, Tesla, Ford
                        - Years covered: 2021-2023
                        - LLM Provider: {st.session_state.model_provider.title()}
            """)

        except Exception as e:
            st.error(f"Initialization error: {str(e)}.")
            st.info("Please check your API keys and data folder structure.")
            st.stop()


def display_sample_questions():
    """Display sample questions in sidebar"""

    st.sidebar.header("Sample Questions")

    sample_questions = [
        "What was BMW's total revenue in 2023?",
        "How much revenue did Tesla generate in 2023?",
        "What was Ford's revenue for 2021?",
        "What were Tesla's profit numbers for 2022 and 2023?",
        "Between Tesla and Ford, which company achieved higher profits in 2022?",
        "What key economic factors influenced Ford's performance in 2021?",
        "Provide a summary of revenue figures for Tesla, BMW, and Ford over the past three years.",
        "What were BMW's profit figures for 2021 and 2023?",
        "Which company recorded better profitability in 2022 overall?"
    ]

    st.sidebar.markdown("**Click to ask:**")
    for i, q in enumerate(sample_questions[:6]):  # Show first 6
        if st.sidebar.button(f"{q}", key=f"sample_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            st.rerun()

    with st.sidebar.expander("**More questions**"):
        for i, q in enumerate(sample_questions[6:], 6):
            if st.sidebar.button(f"{q}", key=f"sample_{i}", use_container_width=True):
                st.session_state.messages.append({"role": "user", "content": q})
                st.rerun()


########## MAIN ##########

def main():

    # Sidebar
    with st.sidebar:

        st.header("System Configuration")

        # Display model info
        if st.session_state.model_provider:
            provider_names = {
                "groq": "Groq (Llama 3.3 70B)",
                "huggingface": "HuggingFace (Llama 3.3 70B)"
            }
            st.info(f"**LLM:** {provider_names.get(st.session_state.model_provider, 'Unknown')}.")
            st.info("**Embeddings:** sentence-transformers/all-MiniLM-L6-v2")

            # Show GPU info if initialized
            if st.session_state.initialized and st.session_state.rag_engine:
                try:
                    import torch
                    if torch.cuda.is_available():
                        st.success(f"**GPU:** {torch.cuda.get_device_name(0)}")
                    else:
                        st.info("**Device:** CPU")
                except:
                    pass

            st.info("**Vector Store:** FAISS")

        st.markdown("---")

        # Initialize/Reset button
        if not st.session_state.initialized:
            if st.button("Initialize System", use_container_width=True, type="primary"):
                initialize_rag_system()
        else:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("Reset", use_container_width=True):
                    st.session_state.initialized = False
                    st.session_state.messages = []
                    if st.session_state.rag_engine:
                        st.session_state.rag_engine.clear_memory()
                    st.rerun()
            with col2:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    if st.session_state.rag_engine:
                        st.session_state.rag_engine.clear_memory()
                    st.rerun()

        st.markdown("---")

        # Sample questions
        if st.session_state.initialized:
            display_sample_questions()

        st.markdown("---")

        # About section
        st.markdown("### Dataset")
        st.markdown("""
        - **BMW**: 2021, 2022, 2023
        - **Tesla**: 2022, 2023
        - **Ford**: 2021, 2022, 2023
        """)

        st.markdown("### Tech Stack")
        st.markdown("""
        - LangChain
        - FAISS Vector Store
        - Sentence Transformers
        - LLM APIs
        """)

        st.markdown("---")
        st.markdown("*Built for Consigli-Ai Technical Interview*")

    # Main content area
    if not st.session_state.initialized:
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div class="info-box">
                <h2 style="text-align: center;">Welcome!</h2>
                <p style="text-align: center;">
                    This is a RAG-based financial analysis system for automotive companies.
                </p>
                <p style="text-align: center;">
                    Click <strong>"Initialize System"</strong> in the sidebar to begin.
                </p>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("""
            <div class="header-container">
            <h1 class="main-header">Automotive Financial Analyst</h1>
            <div class="subtitle">RAG-based Q&A System for BMW, Tesla & Ford Annual Reports (2021-2023)</div>
            </div>
            """, unsafe_allow_html=True)
            st.markdown("### Features")
            st.markdown("""
            - **Conversational Interface**: Ask follow-up questions naturally
            - **Multi-Company Analysis**: Compare BMW, Tesla, and Ford
            - **Time-Series Data**: Analyze trends from 2021-2023
            - **Source Citations**: See exact report references
            - **Accurate Answers**: Powered by advanced RAG pipeline
            """)

            st.markdown("### What You Can Ask")
            st.markdown("""
            - Financial metrics (revenue, profit, EBITDA)
            - Year-over-year comparisons
            - Company performance rankings
            - Economic factors and trends
            - Product information
            """)
        return

    # Chat interface
    st.markdown("---")

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            # Display sources if present
            if "sources" in message:
                with st.expander(" View Sources"):
                    for i, source in enumerate(message["sources"], 1):
                        st.markdown(f"**Source {i}:** {source['company']} {source['year']} - {source['filename']}")
                        st.text(source['content'][:200] + ".")

    # Chat input
    if prompt := st.chat_input("Ask a question about the automotive reports.", key="chat_input"):
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Analyzing documents and generating answer, please wait."):
                try:
                    response = st.session_state.rag_engine.query(
                        prompt,
                        chat_history=st.session_state.messages[:-1]
                    )

                    answer = response["answer"]
                    st.markdown(answer)

                    # Show sources
                    source_docs = response.get("source_documents", [])
                    if source_docs:
                        with st.expander(f"Sources ({len(source_docs)} documents)"):
                            for i, doc in enumerate(source_docs, 1):
                                metadata = doc.metadata
                                st.markdown(f"""
                                **Source {i}:**  
                                Company: {metadata.get('company', 'Unknown')}  
                                Year: {metadata.get('year', 'Unknown')}  
                                File: {metadata.get('filename', 'Unknown')}
                                """)
                                st.text(doc.page_content[:300] + ".")
                                st.markdown("---")

                        # Prepare sources for storage
                        sources = [
                            {
                                "company": doc.metadata.get('company', 'Unknown'),
                                "year": doc.metadata.get('year', 'Unknown'),
                                "filename": doc.metadata.get('filename', 'Unknown'),
                                "content": doc.page_content
                            }
                            for doc in source_docs
                        ]
                    else:
                        sources = []

                    # Add assistant response to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    error_msg = f"Error: {str(e)}"
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })


if __name__ == "__main__":
    main()
