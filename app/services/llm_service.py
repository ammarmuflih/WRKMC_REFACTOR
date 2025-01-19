from typing import Dict, Any
from app.services.helper import helper
from app.config import config
import random

# is_chain = {
#     "is_greeting": chainZoo.is_greeting_chain(),
#     "is_water_level_question": chainZoo.is_waterLevel_question_chain(),
#     "is_identity_question": chainZoo.is_identity_question_chain()
# }

class ragService:
    def __init__(self, retriever, rag_chain, url):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.url = url
        self.helper = helper(llm=config.llm_configs['model_2'].model)
        chainZoo = self.helper.getChainZoo()  # Ensure chainZoo is an instance

        self.is_chain = {
            "is_greeting": chainZoo.is_greeting_chain(),
            "is_water_level_question": chainZoo.is_waterLevel_question_chain(),
            "is_identity_question": chainZoo.is_identity_question_chain()
        }
    
    def preprocessQuery(self, query):
        chain_result = {}
        chain_response = {}

        for chain_name, chain in self.is_chain.items():
            response = chain.invoke({"message":query})
            chain_response[chain_name] = response
            print("REAL RESPONSE: ", chain_name, " : ", chain_response[chain_name])
            response_text = response['text']

            if response_text == '':
                response_text = "False"
                chain_result[chain_name] = False
            elif response_text.strip() == "False" or response_text.strip() == "True":
                chain_result[chain_name] = eval(response_text)
            else:
                response_text = "False"
                chain_result[chain_name] = False
        
        return chain_result, chain_response

    def rag(self, user_id:str,  query:str):
        chat_history = self.helper.get_user_chat_history(user_id=user_id)
        chat_history.add_user_message(query)
        chain_result, chain_response = self.preprocessQuery(query)
        if chain_result["is_greeting"]:
            response = self.helper.greeting_response(greeting=query)
            print(response, type)
            return self.response_template(query=query, response=response)
        
        if chain_result["is_water_level_question"]:
            water_data = self.helper.load_water_level_data(self.url)
            titik_pengamatan, titik_pengamatan_processed, titik_pengamatan_dict = self.helper.get_titik_pengamatan(water_data)
            location = self.helper.is_any_location_in_query(query, titik_pengamatan_processed)
            if location:
                response = {'text': self.helper.water_level_question_response(location=location, data=water_data, titik_pengamatan_dict=titik_pengamatan_dict)}
                return self.response_template(query=query, response=response)
            else:
                titik_pengamatan_string = ', '.join(titik_pengamatan)
                response = {'text': f"Lokasi tidak ditemukan, Daftar Lokasi: {titik_pengamatan_string}"}
                return self.response_template(query=query, response=response)
        
        if chain_result["is_identity_question"]:
            response = {'text': self.helper.identity_question_response()}
            print(response, type)
            return self.response_template(query=query, response=response)
        
        retrieved_docs = self.helper.search_with_similarity(retriever=self.retriever, query=query)
        print("RETRIEVED_DOCS: \n", retrieved_docs)

        print("Chat Service")
        response = {'text': "Chat Service"}
        return self.response_template(query=query, response=response)
    
    def response_template(self, query:str, response:str):
        return {
            'input': query,
            'context': [],
            'messages': [],
            'answer': response['text']
        }