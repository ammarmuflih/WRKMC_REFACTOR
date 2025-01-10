"""
Initialization script for the RAG system module.

This module provides functionality for initializing and managing components of a Retrieval-Augmented Generation (RAG) system, 
including document processing, vector store management, and chain configuration.
"""

# from .document_processor import DocumentProcessor, DocumentProcessingConfig
# from .vectorstore_manager import VectorStoreManager
# from .chain_factory import ChainFactory
# from .initializer import Initializer

from .initializer import Initializer
from .initializer import DocumentProcessor
from .initializer import VectorStoreManager
from .initializer import mainChain
from .initializer_store import InitializerStore

Initializer.initialize_env()