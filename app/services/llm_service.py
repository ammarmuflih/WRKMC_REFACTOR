from typing import Dict, Any
from app.services.helper import helper
from app.config import config
import random
import re
from langchain.docstore.document import Document

class ragService:
    def __init__(self, retriever, rag_chain, url):
        self.retriever = retriever
        self.rag_chain = rag_chain
        self.url = url
        self.llm = config.llm_configs['model_2'].model
        self.helper = helper(llm=config.llm_configs['model_1'].model)
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
                response = {'answer': self.helper.water_level_question_response(location=location, data=water_data, titik_pengamatan_dict=titik_pengamatan_dict)}
                return self.response_template(query=query, response=response)
            else:
                titik_pengamatan_string = ', '.join(titik_pengamatan)
                response = {'answer': f"Lokasi tidak ditemukan, Daftar Lokasi: {titik_pengamatan_string}"}
                return self.response_template(query=query, response=response)
        
        if chain_result["is_identity_question"]:
            response = {'answer': self.helper.identity_question_response()}
            print(response, type)
            return self.response_template(query=query, response=response)
        
        retrieved_docs = self.retriever.invoke(query)
        processed_retrieved_docs = self.preprocess_retrieved_docs(retrieved_docs)

        context = [doc.page_content for doc in processed_retrieved_docs]
        print(context)  
        print(type(context)) 

        if(len(processed_retrieved_docs) == 0):
            print("FALL BACK MECHANISM")
            fallback_response = self.llm.invoke(query)
            response = {'answer': fallback_response.content}
            print("FALLBACK RESPONSE: ", response)
            return self.response_template(query=query, response=response)

        response = self.rag_chain.invoke({
            'input': query,
            'context': context,
            'messages': chat_history.messages
        })

        response['answer'] = self.normalize_terms(response['answer'])

        return self.response_template(query=query, response=response)
    
    def response_template(self, query:str, response:str):
        return {
            'input': query,
            'context': [],
            'messages': [],
            'answer': response['answer']
        }
    
    def preprocess_retrieved_docs(self, retrieved_docs: Any):
        result_to_keep = []
        
        for doc in retrieved_docs:
            score_filter = []
            metadata = getattr(doc, "metadata", {})
            sub_docs = metadata.get("sub_docs", [])

            for sub_doc in sub_docs:
                score = sub_doc.metadata.get("score")
                if score is not None:
                    try:
                        score_filter.append(float(score))
                    except ValueError:
                        print(f"Invalid score format: {score}")

            panjang_data = len(score_filter)

            if panjang_data > 0:
                avg = sum(score_filter) / panjang_data
                if avg >= 0.72:  # toleransi kecil tidak terlalu diperlukan di sini
                    result_to_keep.append(doc)
                else:
                    print("Data akan dihapus")
            else:
                print("Tidak ada sub_doc yang tersedia")

        # Menghapus 'sub_docs' dari metadata dokumen yang tersimpan
        preprocessed_docs = []
        for doc in result_to_keep:
            # Flatten metadata and retain only necessary fields
            simplified_metadata = {
                'source': doc.metadata.get('source'),
                'page': doc.metadata.get('page')
            }
            # Create a new document with simplified metadata
            preprocessed_doc = Document(metadata=simplified_metadata, page_content=doc.page_content)
            preprocessed_docs.append(preprocessed_doc)

        # Kembalikan seluruh list preprocessed_docs
        print("Keep : ",result_to_keep)
        return preprocessed_docs

    def normalize_terms(self, text):
        # Daftar istilah yang perlu dinormalisasi
        normalization_rules = {
            r"SIP\s*ASI": "SIPASI",
            r"SIP-ASI": "SIPASI",
        }

        # Lakukan normalisasi untuk setiap aturan
        for pattern, replacement in normalization_rules.items():
            text = re.sub(pattern, replacement, text, flags=re.IGNORECASE)

        # Kasus khusus: Jika "SIP ASI" muncul dalam kutipan, biarkan seperti itu
        text = re.sub(r'"(.*?)SIP ASI(.*?)"', r'"\1SIP ASI\2"', text)

        return text


