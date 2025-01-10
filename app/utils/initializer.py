"""
Module for initializing and managing RAG system components.

This module handles document loading, database initialization, and various chain configurations
for a Retrieval-Augmented Generation (RAG) system.
"""

import os
from dataclasses import dataclass
from typing import List, Dict, Optional, Any
from pathlib import Path
from dotenv import load_dotenv
from tqdm import tqdm
from langchain.chains import LLMChain, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_experimental.text_splitter import SemanticChunker
from app.config import config

@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing parameters."""
    min_chunk_size: int = config.documenProcessing.min_chunk_size
    max_chunk_size: int = config.documenProcessing.max_chunk_size
    batch_size: int = config.documenProcessing.batch_size
    vector_batch_size: int = config.documenProcessing.vector_batch_size


class DocumentProcessor:
    """Handles document loading and processing operations."""
    
    def __init__(self, folder_path: Path, config):
        self.folder_path = Path(folder_path)
        self.config = config
        
    def load_documents(self) -> List[Any]:
        """Load PDF documents from the specified folder."""
        if not self.folder_path.exists():
            raise FileNotFoundError(f"Document folder not found: {self.folder_path}")
            
        loader = DirectoryLoader(
            str(self.folder_path),
            glob="**/*.pdf",
            loader_cls=PyPDFLoader,
            show_progress=True
        )
        return loader.load()

    def process_documents(self, docs: List[Any], embedding_model) -> List[Any]:
        """Process documents into chunks using semantic splitting."""
        text_splitter = SemanticChunker(
            embedding_model,
            breakpoint_threshold_type='interquartile'
        )
        
        splits = []
        for i in range(0, len(docs), DocumentProcessingConfig.batch_size):
            batch = docs[i:i + DocumentProcessingConfig.batch_size]
            with tqdm(total=len(batch), desc=f"Processing batch {i//DocumentProcessingConfig.batch_size + 1}") as pbar:
                for doc in batch:
                    try:
                        doc_splits = text_splitter.split_documents([doc])
                        filtered_splits = [
                            split for split in doc_splits 
                            if DocumentProcessingConfig.min_chunk_size <= len(split.page_content) <= DocumentProcessingConfig.max_chunk_size
                        ]
                        splits.extend(filtered_splits)
                        pbar.update(1)
                    except Exception as e:
                        print(f"Warning: Failed to process document: {str(e)}")
                        continue
        
        if not splits:
            raise ValueError("No valid document chunks were created")
        
        return splits

class VectorStoreManager:
    """Manages vector store operations and retrieval."""
    
    def __init__(self, embedding_model: Any):
        self.embedding_model = embedding_model

    def create_vectorstore(self, splits: List[Any], batch_size: int) -> FAISS:
        """Create FAISS vector store from document splits."""
        vectorstore = None
        for i in range(0, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            if vectorstore is None:
                vectorstore = FAISS.from_documents(
                    documents=batch,
                    embedding=self.embedding_model
                )
            else:
                batch_vectorstore = FAISS.from_documents(
                    documents=batch,
                    embedding=self.embedding_model
                )
                vectorstore.merge_from(batch_vectorstore)
        return vectorstore

    @staticmethod
    def create_retriever(vectorstore: FAISS, k: int = 5) -> Any:
        """Create a retriever from the vector store."""
        return vectorstore.as_retriever(
            search_kwargs={"k": k, "return_scores": True}
        )

class mainChain:
    """Factory class for creating various LLM chains."""
    
    def __init__(self, llm_model: Any):
        self.llm_model = llm_model

    def create_rag_chain(self, retriever: Any) -> Optional[Any]:
        """Create RAG chain with the specified retriever."""
        if not retriever:
            return None
            
        system_prompt = """
        Anda adalah asisten untuk tugas menjawab pertanyaan.
        Gunakan potongan konteks yang diambil berikut ini untuk memberikan jawaban yang komprehensif dan mendalam.
        Jawab dengan bahasa Indonesia.
        Jika ditanya siapa kamu, jelaskan bahwa kamu adalah asisten virtual WRKMC dengan bahasa Anda sendiri.
        Jika pertanyaan ambigu cukup jawab singkat tanpa menyebutkan isi potongan konteks

        Konteks:
        {context}
        """

        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(self.llm_model, prompt)
        return create_retrieval_chain(retriever, question_answer_chain)

    def create_water_level_chain(self, time: Optional[str] = None, 
                               location: Optional[str] = None, 
                               water_level: Optional[float] = None) -> LLMChain:
        """Create chain for water level reporting."""
        system_prompt = """
        Anda adalah asisten untuk menjawab data mengenai ketinggian muka air atau ketinggian air pada suatu daerah.
        Anda menjawab berdasarkan data yang diberikan. Jawablah secara lengkap, termasuk hari, tanggal, tahun, jam, dan lokasi pengamatan.
        Gunakan kalimat yang Netral dan Objektif

        {context}
        """

        context_parts = []
        if time:
            context_parts.append(f"time: {time}")
        if location:
            context_parts.append(f"titik pengamatan: {location}")
        if water_level is not None:
            context_parts.append(f"water level: {water_level}")

        combined_context = "\n".join(context_parts)

        prompt = PromptTemplate(
            input_variables=["context"],
            template=system_prompt
        )

        return LLMChain(llm=self.llm_model, prompt=prompt)


class Initializer:
    """Main initializer class for the RAG system."""
    
    def __init__(self, folder_path: str):
        self.embedding_model = config.embedding_model_2
        self.llm_model = config.llm_configs['model_1'].model
        self.folder_path = Path(folder_path)
        self.doc_config = DocumentProcessingConfig()
        self.doc_processor = DocumentProcessor(self.folder_path, self.doc_config)
        self.vector_manager = VectorStoreManager(embedding_model=self.embedding_model)
        self.chain_factory = mainChain(llm_model=self.llm_model)

    @staticmethod
    def initialize_env() -> None:
        """Initialize environment variables."""
        load_dotenv()
        
        env_vars = {
            "GOOGLE_API_KEY": os.getenv("GOOGLE_API_KEY"),
            "LANGCHAIN_API_KEY": os.getenv("LANGCHAIN_API_KEY"),
            "LANGCHAIN_TRACING_V2": os.getenv("LANGCHAIN_TRACING_V2", "true")
        }
        
        for key, value in env_vars.items():
            if value and key not in os.environ:
                os.environ[key] = value

    def initialize_database(self) -> Any:
        """Initialize the document database and retriever."""
        try:
            docs = self.doc_processor.load_documents()
            print(f"Loaded {len(docs)} documents")
            
            splits = self.doc_processor.process_documents(docs, self.embedding_model)
            print(f"Created {len(splits)} document chunks")
            
            print("Creating embeddings and building FAISS vectorstore...")
            vectorstore = self.vector_manager.create_vectorstore(
                splits, 
                self.doc_config.vector_batch_size
            )
            
            retriever = self.vector_manager.create_retriever(vectorstore)
            print("Database initialization completed successfully.")
            
            return retriever
            
        except Exception as e:
            print(f"Error during database initialization: {str(e)}")
            raise

    # @staticmethod
    # def load_water_level_data(config, url: str) -> pd.DataFrame:
    #     """Load and process water level data from URL."""
    #     response = requests.get(url)
    #     soup = BeautifulSoup(response.content, 'html.parser')
        
    #     table = soup.find('table')
    #     df = pd.read_html(str(table))[0]
        
    #     # Drop unnecessary columns
    #     columns_to_drop = ['no.'] + [f'C{i}' for i in range(15) if i != 13] + ['ip']
    #     df = df.drop(columns=columns_to_drop)
        
    #     # Rename columns
    #     df = df.rename(columns={
    #         config.water_level.level_column: config.water_level.level_column_rename,
    #         config.water_level.userkey_column: config.water_level.userkey_rename
    #     })
        
    #     # Clean user key data
    #     df[config.water_level.userkey_rename] = (df[config.water_level.userkey_rename]
    #         .str.replace(r'\b(AWLR|AWLMS|AWS)\b', '', regex=True)
    #         .str.strip())
        
    #     return df.reset_index(drop=True)


# Additional chain creation methods can be moved to a separate ChainFactory class if needed
    # def process_documents(self):
    #     try:
    #         folder_path = self.folder_path
    #         if not os.path.exists(folder_path):
    #             raise FileNotFoundError(f"Document folder not found: {folder_path}")
            
    #         docs = DocumentProcessor.load_documents(self)
    #         print(f"Loaded {len(docs)} documents")

    #         # 2. Split documents
    #         print("Processing documents...")
    #         text_splitter = SemanticChunker(
    #             self.config.embedding_model_2,
    #             breakpoint_threshold_type='interquartile'
    #         )

    #         # Batch process documents for better memory management
    #         splits = []
    #         batch_size = 10
    #         for i in range(0, len(docs), batch_size):
    #             batch = docs[i:i + batch_size]
    #             with tqdm(total=len(batch), desc=f"Processing batch {i//batch_size + 1}") as pbar:
    #                 for doc in batch:
    #                     try:
    #                         doc_splits = text_splitter.split_documents([doc])
    #                         # Filter splits yang terlalu pendek atau terlalu panjang
    #                         filtered_splits = [
    #                             split for split in doc_splits 
    #                             if 100 <= len(split.page_content) <= 2000
    #                         ]
    #                         splits.extend(filtered_splits)
    #                         pbar.update(1)
    #                     except Exception as e:
    #                         print(f"Warning: Failed to process document: {str(e)}")
    #                         continue
            
    #         if not splits:
    #             raise ValueError("No valid document chunks were created")
                
    #         print(f"Created {len(splits)} document chunks")
            
    #         # 3. Create embeddings and vectorstore
    #         print("Creating embeddings and building FAISS vectorstore...")
    #         embedding = self.config.embedding_model_2
            
    #         # Process in batches to manage memory
    #         vectorstore_pdf = None
    #         batch_size = 100
    #         for i in range(0, len(splits), batch_size):
    #             batch = splits[i:i + batch_size]
    #             if vectorstore_pdf is None:
    #                 vectorstore_pdf = FAISS.from_documents(
    #                     documents=batch,
    #                     embedding=embedding
    #                 )
    #             else:
    #                 batch_vectorstore = FAISS.from_documents(
    #                     documents=batch,
    #                     embedding=embedding
    #                 )
    #                 vectorstore_pdf.merge_from(batch_vectorstore)
                
    #         retriever = vectorstore_pdf.as_retriever(
    #             search_kwargs={"k": 5, "return_scores": True}
    #         )
            
    #         print("Database initialization completed successfully.")
    #         return retriever
    #     except:
    #         print(f"Error during database initialization: {str(e)}")
    #         raise