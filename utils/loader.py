"""
loader.py
Handles loading of uploaded insurance policy PDFs into LangChain Document objects.
"""

import os
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def save_uploaded_files(uploaded_files, save_dir: str = "data/uploads") -> List[str]:
    """Persist Streamlit UploadedFile objects to disk and return their paths."""
    os.makedirs(save_dir, exist_ok=True)
    saved_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(save_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_paths.append(file_path)
    return saved_paths


def load_pdfs(file_paths: List[str]) -> List[Document]:
    """Load one or more PDF files into a flat list of LangChain Documents."""
    all_docs: List[Document] = []
    for path in file_paths:
        loader = PyPDFLoader(path)
        docs = loader.load()
        for doc in docs:
            doc.metadata["source"] = os.path.basename(path)
        all_docs.extend(docs)
    return all_docs
