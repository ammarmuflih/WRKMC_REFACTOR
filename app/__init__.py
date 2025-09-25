import os
import shutil
from flask import Flask
from flask_cors import CORS
from app.config import config
from app.utils import Initializer
from app.utils import InitializerStore

from app.services import ragService

import os

def precheck():
    base_database_dir = config.paths.public_location_folder
    # Hapus folder yang tidak ada dalam listLokasi
    existing_folders = set(os.listdir(base_database_dir))  # Set folder yang ada di direktori
    allowed_folders = {loc["name"] for loc in config.location_config()}  # Set folder yang seharusnya ada

    print(f"\nFound {len(allowed_folders)} location in configuration\n")
    for loc in allowed_folders:
        print("> ",loc)

    # check if directory is ready and at least there's 1 pdf file
    print("\nChecking database folder... \U0001F9D0")
    for loc in config.location_config():
        file_path = loc["path"]
        if not os.path.exists(file_path):
            print(f"{loc['name']} databse folder not found!")
            print(f"make {loc['name']} database folder")
            os.makedirs(file_path)
        else:
            print(f"> {loc['name']} folder already exist")

    # Cari folder yang tidak ada dalam listLokasi
    for folder in existing_folders - allowed_folders:
        folder_path = os.path.join(base_database_dir, folder)
        if os.path.isdir(folder_path):
            shutil.rmtree(folder_path)  # Hapus folder
            print(f"> Folder '{folder_path}' deleted.")

    for loc in config.location_config():
        file_path = loc["path"]
        if not os.path.exists(file_path):
            print(f"Failed to make {loc['name']}")
            return {"error": f"Failed to make {loc['name']}"}
        else:
            print(f"> {loc['name']} \U00002705")
    print("Checking database folder done \U00002705")
    print("All checks passed. \U00002705")
    return create_app()  # Return the Flask app only if all checks pass


def create_app():
    app = Flask(__name__)

    # CORS(app, resources={
    #     r"/*": {
    #         "origins": ["https://kedungputri.wrkmc-ugm.id", "https://sapon.wrkmc-ugm.id", "https://api.wrkmc-ugm.id"],
    #         "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    #         "allow_headers": ["Content-Type", "Authorization"],
    #         "supports_credentials": True
    #     }
    # })

    initializer_store = InitializerStore()
    for loc in config.location_config():
        initializer = Initializer(folder_path=loc["path"])
        initializer.initialize_env()
        retriever = initializer.initialize_database()
        chain_factory = initializer.chain_factory
        rag_chain = chain_factory.create_rag_chain(retriever=retriever)
        llm_service = ragService(retriever=retriever, rag_chain=rag_chain, url=loc["url"])

        # initializer.load_water_level_data(config=config, url=loc["url"])
        initializer_store.initializer_dict[loc['name']] = {
            "retriever": retriever,
            "rag_chain": rag_chain,
            "llm_service": llm_service,
        }

    from app.routes import api
    app.register_blueprint(api.bp)

    return app

# OLD CODE
    # change database filename to lower and change space to underscore
    # print("\nPreprocessing Documents... \U0001F9D0")
    # for loc in config.location_config():
    #     for filename in loc['path']:
    #         old_file_path = os.path.join(loc['path'], filename)
            
    #         if os.path.isfile(old_file_path):
    #             new_filename = filename.lower().replace(" ","_")
    #             new_file_path = os.path.join(loc['path'], new_filename)

    #             if old_file_path != new_file_path:
    #                 os.rename(old_file_path, new_file_path)
    #                 print(f"Renamed: {filename} -> {new_filename}")
    #             else:
    #                 print(f"File {filename} sudah sesuai format.")

    # print("Preprocessing Documents Done \U00002705")