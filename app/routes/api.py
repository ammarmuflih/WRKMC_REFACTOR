# app/routes/api.py
from flask import Blueprint, request, jsonify
from app.utils import InitializerStore
# from app.services.rag_service import RAGService

bp = Blueprint('api', __name__)
initializer_store = InitializerStore()

@bp.route('/chat', methods=['POST'])
def chat():
    user_id = request.json.get('user_id')
    user_prompt = request.json.get('message') # 'user_prompt'
    user_location = request.json.get('location') # 'user_location'

    print("OBJECT: \n",initializer_store.initializer_dict)

    if user_location == 'sapon':
        response = initializer_store.initializer_dict[user_location]['llm_service'].rag(user_id, user_prompt)
        print("RESPONSE: \n")
        print(response)
        return response
        
    elif user_location == 'kedungputri':
        response = initializer_store.initializer_dict[user_location]['llm_service'].rag(user_id, user_prompt)
        print("RESPONSE: \n")
        print(response)
        return response