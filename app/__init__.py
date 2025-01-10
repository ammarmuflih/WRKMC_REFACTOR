from flask import Flask
from app.config import config
from app.utils import Initializer
from app.utils import InitializerStore

from app.services import ragService

def create_app():
    app = Flask(__name__)

    initializer_store = InitializerStore()
    for loc in config.location_config():
        initializer = Initializer(folder_path=loc["path"])
        initializer.initialize_env()
        retriever = initializer.initialize_database()
        rag_chain = initializer.chain_factory
        llm_service = ragService(retriever=retriever, rag_chain=rag_chain)

        # initializer.load_water_level_data(config=config, url=loc["url"])
        initializer_store.initializer_dict[loc["name"]] = {
            "retriever": retriever,
            "rag_chain": rag_chain,
            "llm_service": llm_service,
        }

    from app.routes import api
    app.register_blueprint(api.bp)

    return app

# old code

# def create_app(config_class = config):
#     app = Flask(__name__)
#     # app.config.from_object(config_class)

#     # Initialize RAG
#     # Initializer.initializeRAG

#     # Register blueprints
#     # from app.routes import api

# saponDocumentProcessor = DocumentProcessor(folder_path=config.paths.sapon, config=config)
# kedungputriDocumentProcessor = DocumentProcessor(folder_path=config.paths.kedungputri, config=config)

# saponVectorStoreManager = VectorStoreManager(embedding_model=config.embedding_model_2)
# kedungPutriVectoreStoreManager = VectorStoreManager(embedding_model=config.embedding_model_2)

# saponProcessedDocuments = saponDocumentProcessor.process_documents()
# kedungPutriProcessedDocuments = kedungputriDocumentProcessor.process_documents()

# saponVectoreStore = saponVectorStoreManager.create_vectorstore(splits=saponProcessedDocuments, batch_size=10)
# kedungPutriVectoreStore = kedungPutriVectoreStoreManager.create_vectorstore(splits=kedungPutriProcessedDocuments, batch_size=10)

# saponRetriever = saponVectorStoreManager.create_retriever()
# kedungPutriRetriever = kedungPutriVectoreStoreManager.create_retriever()

# === #

# saponInitializer = Initializer(folder_path=config.paths.sapon, config=config)
# kedungPutriInitializer = Initializer(folder_path=config.paths.kedungputri, config=config)

# saponInitializer.initialize_env()
# saponInitializer.initialize_database()

# kedungPutriInitializer.initialize_env()
# kedungPutriInitializer.initialize_database()

# saponInitializer.load_water_level_data(config=config, url=config.water_level.sapon_url)
# kedungPutriInitializer.load_water_level_data(config=config, url=config.water_level.kedungputri_url)
