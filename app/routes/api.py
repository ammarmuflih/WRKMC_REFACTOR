# app/routes/api.py
from flask import Blueprint, request, jsonify
from app.utils import InitializerStore
from app.config import config
from app.database import databaseHandler
from app.services.helper import helper
import os
# from app.services.rag_service import RAGService

initializer_store = InitializerStore()
print(initializer_store.initializer_dict['sapon']['retriever'])
Helper = helper(llm=config.llm_configs['model_2'].model)

# ANSI escape codes (Move this code to other new module "style")
BOLD = '\033[1m'
END = '\033[0m'
BACKGROUND = '\x1b[47m'  # Warna latar belakang (highlight) kuning
FOREGROUND = '\x1b[30m'
RESET = '\033[0m'  # Reset formatting

branch_info = config.location_config()
for loc in branch_info:  # Jika branch_info adalah list, iterasi langsung
    location = loc["name"]  # Ambil nilai "name" dari setiap dictionary dalam list
    
    item = initializer_store.initializer_dict[location].keys()  # Ambil semua key
    print(f"{BOLD}{BACKGROUND}{FOREGROUND}========================================{RESET}")
    print(f"{BOLD}{BACKGROUND}{FOREGROUND}           Branch: {location}           {RESET}")
    print(f"{BOLD}{BACKGROUND}{FOREGROUND}========================================{RESET}")
    
    for key in item:
        print(f"{BOLD}{key}:{RESET}")
        
        # Mendapatkan nilai untuk setiap key
        value = initializer_store.initializer_dict[location][key]
        
        # Jika nilai berupa objek atau struktur data yang kompleks, buat format yang lebih baik
        if isinstance(value, str):
            print(f"    {value}")
        elif isinstance(value, dict) or isinstance(value, list):
            print(f"    {value.__class__.__name__} - {str(value)[:100]}...")  # Tampilkan class type dan potongan pertama
        else:
            print(f"    {value}")
        
        print(f"{BOLD}-" * 40 + f"{RESET}")  # Pemisah antar key dengan format tebal

    print(f"{BOLD}========================================{RESET}\n")

bp = Blueprint('api', __name__)

@bp.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')
    user_prompt = request.json.get('message') # 'user_prompt'
    user_location = request.json.get('location') # 'user_location'

    if user_location in initializer_store.initializer_dict:
        response = initializer_store.initializer_dict[user_location]['llm_service'].rag(user_id, user_prompt)
        print("RESPONSE: \n")
        print(response)
        return response
    else:
        return {"error":"Location not supported"}, 400

@bp.route('/preAddToDatabase', methods=['POST'])
def preAddToDataBase():
    location = request.form.get('location')
    file = request.files['file']
    fileName = file.filename

    if fileName == '':
        return jsonify({"error": "No file part in the request"}), 400

    for loc in config.location_config():
        if loc["name"] == location:
            file_check = os.path.join(loc["path"], fileName)
            if os.path.exists(file_check):
                return {"error": f"file {file.filename} already at the directory"}
            else:
                try:
                    databaseHandler.add_to_database(path=loc["path"], file=file)
                    retriever, rag_chain, llm_service = databaseHandler.refresh_retriever(location=location)
                    initializer_store.initializer_dict[location]['retriever'] = retriever
                    initializer_store.initializer_dict[location]['rag_chain'] = rag_chain
                    initializer_store.initializer_dict[location]['llm_service'] = llm_service
                    return {"success": f"Added '{file.filename}' to '{loc['name']}' database folder"}
                except:
                    return {"error": f"Failed to add '{fileName}' to '{loc['name']}' database folder"}

@bp.route('/preDeleteFromDatabase', methods=['POST'])
def preDeletedDFromDatabase():
    location = request.form.get('location')
    file = request.files.get('file')

    if not file or file.filename == '':
        return jsonify({"error": "Input not valid"})
    
    fileName = file.filename

    for loc in config.location_config():
        if loc["name"] == location:
            file_check = os.path.join(loc["path"], fileName)
            if not os.path.isfile(file_check):  # Gunakan os.path.isfile() agar lebih spesifik
                return jsonify({"error": "File not found"})
            
            # Simpan hasil delete dan kembalikan sebagai response
            result = databaseHandler.delete_from_database(path=loc["path"], file=fileName)
            retriever, rag_chain, llm_service = databaseHandler.refresh_retriever(location=location)
            initializer_store.initializer_dict[location]['retriever'] = retriever
            initializer_store.initializer_dict[location]['rag_chain'] = rag_chain
            initializer_store.initializer_dict[location]['llm_service'] = llm_service
            return jsonify(result)

    return jsonify({"error": "Location not found"})

@bp.route('/refreshRetriever',  methods=['POST'])
def refreshRetriever():
    location = request.form.get('location')

    for loc in config.location_config():
        if loc["name"] == location:
            try:
                retriever, rag_chain, llm_service = databaseHandler.refresh_retriever(location=location)
                initializer_store.initializer_dict[location]['retriever'] = retriever
                initializer_store.initializer_dict[location]['rag_chain'] = rag_chain
                initializer_store.initializer_dict[location]['llm_service'] = llm_service
                return {"success": f"'{location}' retriever has been updated"}
            except:
                return {"error": f"Failed to update '{location} retriever'"}

@bp.route('/summarize', methods=['POST'])
def summarizeText():
    location = request.json.get('location')
    text = request.json.get('text')
    summarized_text = Helper.summarize(text=text)
    return {"summarized_text": summarized_text['text']}
    



