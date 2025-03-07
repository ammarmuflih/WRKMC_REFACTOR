import os
from app.config import config
from app.utils import Initializer
from app.services import ragService


class databaseHandler:
    def __init__(self):
        pass
    
    @staticmethod
    def add_to_database(path=None, file=None):
        # Cek apakah path tidak None
        if path is not None:
            # Cek apakah file tidak None
            if file is not None:
                # Periksa apakah path adalah direktori yang valid
                if not os.path.isdir(path):
                    return {"error": f"The provided path '{path}' is not a valid directory."}

                # Gabungkan path dan nama file
                file_path = os.path.join(path, file.filename)

                try:
                    # Simpan file ke path tujuan
                    file.save(file_path)
                    return {"success": f"File '{file.filename}' successfully uploaded to '{path}'."}
                except Exception as e:
                    return {"error": f"Failed to save file: {str(e)}"}
            else:
                return {"error": "There's no file to upload"}
        else:
            return {"error": "There's no path destination"}
    
    @staticmethod
    def delete_from_database(path=None, file=None):
        if path is None:
            return {"error": "Database path is needed"}
        if file is None:
            return {"error": "File is needed"}
        
        file_path = os.path.join(path, file)  # Gunakan file sebagai string
        if os.path.isfile(file_path):
            try:
                os.remove(file_path)  # Gunakan os.remove()
                return {"success": f"File '{file}' successfully removed from '{path}'"}
            except Exception as e:
                return {"error": f"Failed to remove '{file}' from '{path}': {str(e)}"}
        else:
            return {"error": "File not found or is a directory"}

    @staticmethod
    def refresh_retriever(location=None):
        if location is None:
            return {"error": "Location field is needed"}

        for loc in config.location_config():
            if loc["name"] == location:
                try:
                    initializer = Initializer(folder_path=loc["path"])
                    retriever = initializer.initialize_database()
                    chain_factory = initializer.chain_factory
                    rag_chain = chain_factory.create_rag_chain(retriever=retriever)
                    llm_service = ragService(retriever=retriever, rag_chain=rag_chain, url=loc['url'])
                    return retriever, rag_chain, llm_service  # Tidak perlu `del initializer`, Python akan menghapusnya otomatis
                except Exception as e:
                    return {"error": f"Failed to initialize database: {str(e)}"}
        
        return {"error": "Location not found"}
