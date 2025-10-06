# Consigli_2/src/vector_store.py


########## IMPORTS ##########

# Standard Library Imports
import logging
from typing import List

# Third-party Imports
import torch
import faiss
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# Initialise logger
logger = logging.getLogger(__name__)


########## CLASSES ##########

class VectorStoreManager:

    """Manages vector store operations with FAISS"""
    
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize with HuggingFace embeddings
        Automatically detects and uses GPU if available
        """

        # Detect GPU availability
        self.device = self._detect_device()
        logger.info(f"Using device: {self.device}")
        
        # Initialize embeddings with GPU support
        self.embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={'device': self.device},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        self.vector_store = None
        self.use_gpu_index = self._check_faiss_gpu()
    
    def _detect_device(self) -> str:
        """Detect if CUDA GPU is available"""

        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU detected: {gpu_name}")
            return 'cuda'
        else:
            logger.info("No GPU detected, using CPU")
            return 'cpu'
    
    def _check_faiss_gpu(self) -> bool:
        """Check if FAISS-GPU is available"""

        try:
            if hasattr(faiss, 'StandardGpuResources'):
                logger.info("FAISS-GPU available")
                return True
            else:
                logger.info("FAISS-CPU mode (install faiss-gpu for GPU acceleration)")
                return False
        except Exception as e:
            logger.error(f"FAISS-CPU mode: {e}")
            return False
    
    def create_index(self, documents: List[Document]):
        """Create FAISS index from documents with GPU acceleration"""

        if not documents:
            raise ValueError("No documents provided for indexing")
        
        logger.info(f"Creating vector store with {len(documents)} documents.")
        
        # Create the FAISS index
        self.vector_store = FAISS.from_documents(documents, self.embeddings)
        
        # If GPU is available and FAISS-GPU is installed, move index to GPU
        if self.use_gpu_index and self.device == 'cuda':
            self._move_index_to_gpu()
        
        logger.info("Vector store created successfully!")
    
    def _move_index_to_gpu(self):
        """Move FAISS index to GPU for faster search"""

        try:
            # Get the current CPU index
            cpu_index = self.vector_store.index
            
            # Create GPU resources
            gpu_resources = faiss.StandardGpuResources()
            
            # Move index to GPU
            gpu_index = faiss.index_cpu_to_gpu(gpu_resources, 0, cpu_index)
            
            # Replace the CPU index with GPU index
            self.vector_store.index = gpu_index
            
            logger.info("FAISS index moved to GPU")
        except Exception as e:
            logger.info(f"Could not move index to GPU: {e}")
            logger.info("Continuing with CPU index")
    
    def similarity_search(self, query: str, k: int = 4) -> List[Document]:
        """Search for similar documents"""

        if self.vector_store is None:
            raise ValueError("Vector store not initialized. Call create_index first.")
        
        return self.vector_store.similarity_search(query, k=k)
    
    def save_local(self, path: str = "vector_store"):
        """Save vector store to disk"""

        if self.vector_store:
            # GPU indices need to be moved to CPU before saving
            if self.use_gpu_index:
                logger.info("Moving index to CPU for saving.")
                self._move_index_to_cpu()
            
            self.vector_store.save_local(path)
            logger.info(f"Vector store saved to {path}")
    
    def _move_index_to_cpu(self):
        """Move FAISS index back to CPU (for saving)"""

        try:
            if hasattr(self.vector_store.index, 'index'):
                # This is a GPU index wrapper
                cpu_index = faiss.index_gpu_to_cpu(self.vector_store.index)
                self.vector_store.index = cpu_index
        except Exception as e:
            logger.error(f"Could not move index to CPU: {e}")
    
    def load_local(self, path: str = "vector_store"):
        """Load vector store from disk"""
        self.vector_store = FAISS.load_local(
            path, 
            self.embeddings,
            allow_dangerous_deserialization=True)
        
        # Move to GPU if available
        if self.use_gpu_index and self.device == 'cuda':
            self._move_index_to_gpu()

    def get_device_info(self) -> dict:
        """Get information about device being used"""
        info = {
            "device": self.device,
            "faiss_gpu": self.use_gpu_index
        }

        if self.device == 'cuda':
            info["gpu_name"] = torch.cuda.get_device_name(0)
            info["gpu_memory_allocated"] = f"{torch.cuda.memory_allocated(0) / 1024 ** 2:.2f} MB"
            info["gpu_memory_cached"] = f"{torch.cuda.memory_reserved(0) / 1024 ** 2:.2f} MB"

        return info
