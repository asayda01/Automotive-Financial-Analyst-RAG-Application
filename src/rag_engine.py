# Consigli_2/src/rag_engine.py


########## IMPORTS ##########

# Standard Library Imports
import os
from typing import List, Dict, Optional

# Third-party Imports
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain_community.llms import HuggingFaceHub
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain


########## CLASSES ##########

class RAGEngine:
    """RAG engine for question answering"""

    def __init__(self, vector_store_manager, model_provider: str = "groq"):
        """
        Initialize RAG engine

        Args:
            vector_store_manager: Vector store instance
            model_provider: "groq" or "huggingface"
        """
        self.vector_store_manager = vector_store_manager
        self.model_provider = model_provider

        # Initialize LLM based on provider
        self.llm = self._initialize_llm(model_provider)

        # Create custom prompt optimized for financial analysis
        self.qa_prompt = PromptTemplate(
            template="""You are an expert financial analyst specializing in automotive companies.
                        Use the following context from annual reports to answer the question.
                
                        Context: {context}
                
                        Question: {question}
                
                        IMPORTANT INSTRUCTIONS:
                        1. Look through ALL the context carefully for each company mentioned
                        2. For revenue data, search for:
                           - "Total revenue" / "Total revenues"
                           - "Consolidated revenues"
                           - Income Statement or Statement of Operations
                           - Company-wide revenue figures (not just segments)
                        3. If you find revenue segments (Ford Pro, Ford Model e, etc.) but no total:
                           - Add up the segments to get total company revenue
                           - State that you calculated it from segments
                        4. For BMW, prioritize "BMW Group" data over "BMW Finance N.V." subsidiary data
                        5. For Ford, look for consolidated company totals, not just segment data
                        6. Present data in the exact structure requested by the user
                        7. If data is truly not in context for a specific year, state: "Data not found in provided reports for [Company] [Year]"
                        8. Always show your reasoning - explain where you found each figure
                
                        Answer clearly with proper formatting:""",
            input_variables=["context", "question"]
        )

        # Initialize memory for conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        # Create retrieval chain with MMR for diversity
        self.qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.vector_store_manager.vector_store.as_retriever(
                search_type="mmr",  # Use MMR
                search_kwargs={
                    "k": 15,  # Retrieve chunks (to reduce memory usage, decrease number )
                    "fetch_k": 30,  # Fetch 30, return best 15
                    "lambda_mult": 0.7  # Balance relevance vs diversity ( maybe 0.5 ? less diversityy ? )
                }
            ),
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": self.qa_prompt},
            verbose=False
        )

    def _initialize_llm(self, provider: str):
        """Initialize LLM based on provider"""

        if provider == "groq":
            # Initialize Groq as provider first
            api_key = os.getenv("GROQ_API_KEY")
            if not api_key:
                raise ValueError("GROQ_API_KEY not found in environment variables")

            # Use current Groq models
            return ChatGroq(
                model="llama-3.3-70b-versatile",  # For more GroQ token access : "llama-3.1-8b-instant"
                temperature=0,
                groq_api_key=api_key,
                max_tokens=1024
            )

        elif provider == "huggingface":
            # HuggingFace Inference API
            api_key = os.getenv("HF_API_KEY")
            if not api_key:
                raise ValueError("HF_API_KEY not found in environment variables")

            return HuggingFaceHub(
                repo_id="meta-llama/Llama-3.1-8B-Instruct",
                huggingfacehub_api_token=api_key,
                model_kwargs={
                    "temperature": 0.1,
                    "max_new_tokens": 512,
                    "top_p": 0.95,
                    "repetition_penalty": 1.1
                }
            )

        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def query(self, question: str, chat_history: Optional[List[Dict]] = None) -> Dict:
        """
        Query the RAG system with improved query handling

        Args:
            question: User question
            chat_history: Previous conversation history

        Returns:
            Dict with 'answer' and 'source_documents'
        """
        try:
            # Enhance query with financial keywords if needed
            enhanced_question = self._enhance_financial_query(question)

            # Format chat history for LangChain
            formatted_history = []
            if chat_history:
                # Use last 6 messages (3 exchanges) to manage context
                for msg in chat_history[-6:]:
                    if msg["role"] == "user":
                        formatted_history.append(("human", msg["content"]))
                    elif msg["role"] == "assistant":
                        formatted_history.append(("ai", msg["content"]))

            # Query chain with enhanced question
            result = self.qa_chain({"question": enhanced_question, "chat_history": formatted_history})

            return {
                "answer": result["answer"],
                "source_documents": result.get("source_documents", [])
            }

        except Exception as e:
            error_msg = str(e)

            if "rate limit" in error_msg.lower():
                return {
                    "answer": "Rate limit reached. Please wait a moment and try again.",
                    "source_documents": []
                }
            elif "timeout" in error_msg.lower():
                return {
                    "answer": "Request timed out. Please try again.",
                    "source_documents": []
                }
            else:
                return {
                    "answer": f"Error: {error_msg}\n\nPlease try rephrasing your question.",
                    "source_documents": []
                }

    def _enhance_financial_query(self, question: str) -> str:
        """Add extensive financial keywords to improve retrieval"""

        question_lower = question.lower()

        enhancements = []

        # Revenue-specific enhancements
        if "revenue" in question_lower or "summary" in question_lower:
            enhancements.append("total revenue revenues consolidated revenue")
            enhancements.append("total revenues net sales company revenue")
            enhancements.append("income statement consolidated statement financial")
            enhancements.append("automotive revenue total company revenue")

            # Company-specific terms
            if "ford" in question_lower:
                enhancements.append("Ford Motor Company consolidated revenues")
                enhancements.append("company total revenue wholesale retail")
            if "bmw" in question_lower:
                enhancements.append("BMW Group revenues consolidated automotive")
                enhancements.append("group revenues total BMW AG")
            if "tesla" in question_lower:
                enhancements.append("Tesla Inc total revenues automotive")

        # Profit/profitability enhancements
        if "profit" in question_lower:
            enhancements.append("net income profit loss operating income")
            enhancements.append("EBIT EBITDA earnings profitability")

        # Multi-year summary requests
        if "summary" in question_lower or "over" in question_lower:
            enhancements.append("financial highlights consolidated results")
            enhancements.append("year ended December 31 fiscal year")
            enhancements.append("income statement statement of operations")

        # Year-specific
        years = ["2020", "2021", "2022", "2023"]
        for year in years:
            if year in question_lower:
                enhancements.append(f"year {year} fiscal {year} FY{year} ended {year}")

        # Financial statement keywords
        enhancements.append("financial statements consolidated operations")

        # Return significantly enhanced query
        if enhancements:
            enhanced = f"{question} {' '.join(enhancements)}"
            return enhanced[:500]  # Limit to 500 chars to avoid too long
        return question

    def clear_memory(self):
        """Clear conversation memory"""
        self.memory.clear()
