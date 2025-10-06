# Consigli_2/src/document_processor.py


########## IMPORTS ##########

# Standard Library Imports
import logging
from typing import List
from pathlib import Path

# Third-party Imports
import PyPDF2
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter


# Initialise logger
logger = logging.getLogger(__name__)


########## CLASSES ##########

class DocumentProcessor:
    """Handles PDF loading and text chunking"""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 400):
        """
        Initialize chunks for financial documents
        """
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def extract_text_from_pdf(self, pdf_path: Path) -> str:
        """Extract text from PDF file"""
        text = ""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            logger.error(f"Error reading {pdf_path}: {e}")
        return text

    def load_and_process_documents(self, data_path: Path) -> List[Document]:
        """Load all PDFs and create document chunks"""
        documents = []

        # Find all PDF files
        pdf_files = list(data_path.rglob("*.pdf"))

        for pdf_path in pdf_files:
            logger.info(f"Processing: {pdf_path}")

            # Extract text
            text = self.extract_text_from_pdf(pdf_path)

            if not text.strip():
                logger.error(f"Warning: No text extracted from {pdf_path}")
                continue

            # Create metadata
            metadata = {
                "source": str(pdf_path),
                "company": pdf_path.parent.name,
                "year": self._extract_year(pdf_path.name),
                "filename": pdf_path.name
            }

            # Split into chunks
            chunks = self.text_splitter.split_text(text)

            # Create Document objects
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk,
                    metadata={**metadata, "chunk": i}
                )
                documents.append(doc)

        return documents

    def _extract_year(self, filename: str) -> str:
        """Extract year from filename"""
        # The provided data files include only 2021,2022 and 2023 papers
        for year in ['2020', '2021', '2022', '2023', '2024', '2025']:
            if year in filename:
                return year
        return "unknown"
