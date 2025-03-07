from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.retrievers.multi_query import MultiQueryRetriever
from typing import Dict, Any
import pandas as pd
import requests
from bs4 import BeautifulSoup
import random
from app.config import config
from typing import List, Dict

class helper:

    def __init__(self, llm):
        self.llm = llm
        self.chainZoo = chainZoo(llm=self.llm)
        self.user_chat_histories: Dict[str, Dict] = {}

    def get_user_chat_history(self, user_id: str) -> ChatMessageHistory:
        if user_id not in self.user_chat_histories:
            self.user_chat_histories[user_id] = {
                "chat_history": ChatMessageHistory(),
                "has_title": False
            }
        
        return self.user_chat_histories[user_id]["chat_history"]
    
    def greeting_response(self, greeting: str):
        greeting_chain = self.chainZoo.greeting_chain()
        response = greeting_chain.invoke({"greeting": greeting})
        return response
    
    def identity_question_response(self) -> str:
        responses = config.responses.random_identity_question_response()
        return responses
    
    def water_level_question_response(self, location, data, titik_pengamatan_dict):
        print("DATA FRAME: \n", data)
        location_data = titik_pengamatan_dict[location[0]]
        filtered_data = data.loc[data['titik pengamatan'] == location_data]

        time = filtered_data['time'].values[0]
        tinggi_muka_air = filtered_data['tinggi muka air'].values[0]

        print("time: ", time)
        print("titik pengamatan: ", location_data)
        print("tinggi muka air: ", tinggi_muka_air)

        water_level_chain = self.chainZoo.water_level_question_chain(time=time, titik_pengamatan=titik_pengamatan_dict, water_level=tinggi_muka_air)
        response = water_level_chain.invoke({"context": f"time: {time}\n lokasi: {location_data}\n tinggi muka air: {tinggi_muka_air} cm"})
        return response

    def load_water_level_data(self, url: str) -> pd.DataFrame:
        """Load and process water level data from URL."""
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        
        table = soup.find('table')
        df = pd.read_html(str(table))[0]
        table = soup.find('table')
        df = pd.read_html(str(table))[0]
        df = df.drop(columns=['no.', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9', 'C10', 'C11', 'C12', 'C14', 'ip'])
        df = df.rename(columns={config.water_level.level_column: config.water_level.level_column_rename})
        df = df.rename(columns={config.water_level.userkey_column: config.water_level.userkey_rename})
        df[config.water_level.userkey_rename] = df[config.water_level.userkey_rename].str.replace(r'\b(AWLR|AWLMS|AWS)\b', '', regex=True).str.strip()
        print("USER_KEY: \n",df[config.water_level.userkey_rename])
        df = df.reset_index(drop=True)
        return df
    
    def is_any_location_in_query(self, message: str, data):
        processed_message = message.lower().replace('.', '').replace('-', '').replace('_', '').replace(' ', '')
        print("Processed Message: ", processed_message)
        is_location = [item for item in data if item in processed_message]
        if is_location:
            print(f"String mengandung karakter: {is_location}")
            return is_location
        else:
            print("Tidak ada karakter yang cocok ditemukan.")
            return False
    
    def get_titik_pengamatan(self, data):
        titik_pengamatan = data['titik pengamatan'].tolist()
        titik_pengamatan_processed = data['titik pengamatan'].str.lower().replace({'\.': '', '-': '', '_': '', ' ': ''}, regex=True).tolist()
        titik_pengamatan_dict = dict(zip(titik_pengamatan_processed, titik_pengamatan))
        return titik_pengamatan, titik_pengamatan_processed, titik_pengamatan_dict
    
    def getChainZoo(self):
        chainZoo = self.chainZoo
        return chainZoo
    
    def summarize(self, text):
        summarize_chain = self.chainZoo.summarize_chain()
        print("INPUT TEXT: ", text)
        summarized_text = summarize_chain.invoke({"message": text})
        return summarized_text

        

class chainZoo:
    ''' 
    using secondary llm for chain
    '''
    def __init__(self, llm):
        self.llm = llm

    def greeting_chain(self):
        llm = self.llm
        prompt = config.promptTemplate.greeting_prompt
        greeting_chain = LLMChain(llm=llm, prompt=prompt)
        return greeting_chain
    
    def water_level_question_chain(self, time, titik_pengamatan, water_level):
        llm = self.llm
        prompt = config.promptTemplate.water_level_question_promp
        question_answer_chain = LLMChain(llm=llm, prompt=prompt)
        return question_answer_chain
    
    def is_greeting_chain(self):
        llm = self.llm
        prompt = config.promptTemplate.is_greeting_prompt

        question_answer_chain = LLMChain(llm=llm, prompt=prompt)
        return question_answer_chain
    
    def is_identity_question_chain(self):
        llm = self.llm
        prompt = config.promptTemplate.is_identity_question_prompt
        question_answer_chain = LLMChain(llm=llm, prompt=prompt)
        return question_answer_chain
        
    def is_waterLevel_question_chain(self):
        llm = self.llm
        prompt = config.promptTemplate.is_waterlevel_question_prompt
        question_answer_chain = LLMChain(llm=llm, prompt=prompt)
        return question_answer_chain
    
    def summarize_chain(self):
        llm = self.llm
        prompt = config.promptTemplate.summarize_prompt
        question_answer_chain =  LLMChain(llm=llm, prompt=prompt)
        return question_answer_chain