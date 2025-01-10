"""Configuration module for the RAG system."""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, List

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
# from llm_zoo import OpenRouterLLM
import dotenv
import os

# Load environment variables
dotenv.load_dotenv()

@dataclass
class DocumentProcessingConfig:
    """Configuration for document processing parameters."""
    min_chunk_size: int = 100
    max_chunk_size: int = 2000
    batch_size: int = 10
    vector_batch_size: int = 100

@dataclass
class PathConfig:
    """Document paths configuration."""
    sapon: Path = Path('/home/ammar/WRKMC/WRKMCLLM_NEW/public_sapon')
    kedungputri: Path = Path('/home/ammar/WRKMC/WRKMCLLM_NEW/public_kedungputri')

@dataclass
class SplitterConfig:
    """Text splitter configuration."""
    chunk_overlap: int = 200
    chunk_size: int = 1000

@dataclass
class WaterLevelConfig:
    """Water level data configuration."""
    kedungputri_url: str = "https://sipasi.online/data/last/di/49"
    sapon_url: str = "https://sipasi.online/data/last/di/47"
    level_column: str = 'C13'
    level_column_rename: str = 'tinggi muka air'
    userkey_column: str = 'user_key'
    userkey_rename: str = 'titik pengamatan'

@dataclass
class LLMConfig:
    """LLM models configuration."""
    name: str
    temperature: float
    model: Any

@dataclass
class promptTemplate:
    greeting_prompt = PromptTemplate(
        input_variables=["greeting"],
        template="Anda diminta untuk merespon sapaan. Pertanyaan: {greeting}. jawab dengan sapaan yang sopan"
    )

    water_level_question_promp = PromptTemplate(
        input_variables=["context"],
        template="Anda adalah asisten untuk menjawab data mengenai ketinggian muka air atau ketinggian air pada suatu daerah. "
            "Anda menjawab berdasarkan data yang diberikan. Jawablah secara lengkap, termasuk hari, tanggal, tahun, jam, dan lokasi pengamatan."
            "gunakan kalimat yang Netral dan Objektif"
            "\n\n"
            "{context}"
    )

    is_greeting_prompt = PromptTemplate(
        input_variables=["message"],
        template='''Tugas: Tentukan apakah input berikut merupakan sapaan, perkenalan diri atau pertanyaan identitas mu.
            Jawab HANYA dengan 'True' atau 'False' tanpa kata tambahan apapun. Abaikan typo pada input

            Definisi:
            1. Sapaan:
            - Kata atau frasa yang digunakan untuk menyapa, memberi salam, atau memulai percakapan
            - Biasanya muncul di awal kalimat atau berdiri sendiri
            - Dapat formal maupun informal

            2. Perkenalan:
            - Kalimat yang menyebutkan nama atau identitas diri
            - Ungkapan yang menjelaskan tentang diri sendiri
            - Dapat mencakup profesi, asal, atau informasi personal lainnya

            Contoh yang menghasilkan True:
            Sapaan:
            - "Halo"
            - "Hi"
            - "Selamat pagi" 
            - "Assalamualaikum"
            - "Permisi"
            - "Hai kawan"
            - "Met pagi"
            - "Pagi semua"
            - "Hey there"
            - "Good morning"

            Perkenalan:
            - "Nama saya John"
            - "Saya Maria dari Jakarta"
            - "Perkenalkan, saya Bobby"
            - "Let me introduce myself"
            - "Saya mahasiswa UI"
            - "I'm the new employee here"
            - "Saya guru di SMA 1"
            - "My name is Jane"
            - "Panggil saya Alex"
            - "Saya tinggal di Bandung"

            Contoh yang menghasilkan False:
            - "Saya mau pesan makanan"
            - "Dimana alamatnya?"
            - "Terima kasih"
            - "Sampai jumpa"
            - "Maaf"
            - "Ok"
            - "Baik"
            - "Sudah"
            - "Ya"
            - "Tidak"

            Pesan yang akan dianalisis: {message}'''
    )

    is_identity_question_prompt = PromptTemplate(
        input_variables=["message"],
        template='''Tugas: Analisis apakah input merupakan pertanyaan tentang identitas AI/asisten.
        Jawab HANYA dengan 'True' atau 'False' tanpa kata tambahan.

        Definisi Pertanyaan Identitas AI/Asisten:
        1. Pertanyaan Langsung tentang Identitas:
        - Menanyakan nama atau identitas ("siapa kamu", "what's your name")
        - Menanyakan jenis/sifat ("apakah kamu AI", "are you a bot")
        - Menanyakan pembuat ("siapa yang membuatmu", "who created you")

        2. Pertanyaan tentang Kemampuan:
        - Menanyakan fungsi spesifik AI ("apa yang bisa kamu lakukan", "what are your capabilities")
        - Menanyakan batasan ("bisakah kamu melihat gambar", "can you remember our chat")
        - Menanyakan cara kerja AI ("bagaimana kamu berpikir", "how do you work")

        3. Pertanyaan tentang Asal-usul:
        - Menanyakan versi ("versi berapa kamu", "which version are you")
        - Menanyakan waktu pembuatan ("kapan kamu dibuat", "when were you created")
        - Menanyakan tujuan pembuatan ("untuk apa kamu dibuat", "why were you created")

        Bukan Pertanyaan Identitas (False):
        1. Permintaan Bantuan Umum:
        - "Tolong bantu saya"
        - "Bisa minta tolong?"
        - "Can you help me?"
        - "Please assist me"

        2. Pertanyaan Tentang Tugas:
        - "Bagaimana cara mengerjakan ini?"
        - "How does this work?"
        - "What should I do?"
        - "Gimana caranya?"

        3. Sapaan dan Kesopanan:
        - "Halo"
        - "Terima kasih"
        - "Selamat pagi"
        - "Good morning"

        4. Respon Singkat:
        - "Ok"
        - "Ya"
        - "Tidak"
        - "Sure"

        5. Pertanyaan Informasi Umum:
        - "Jam berapa sekarang?"
        - "Dimana alamatnya?"
        - "What's the weather like?"
        - "How much is this?"

        Edge Cases (True):
        - "Kamu program apa?" (menanyakan identitas)
        - "Are you virtual WRKMC Assistant?" (menanyakan identitas spesifik)
        - "What kind of assistant are you?" (menanyakan jenis)
        - "Apakah kamu Asisten Virtual WRKMC?" (menanyakan pembuat)

        Edge Cases (False):
        - "Can you explain this?" (permintaan bantuan umum)
        - "Would you help me?" (permintaan bantuan umum)
        - "Is this possible?" (pertanyaan tentang tugas)
        - "Do you know?" (pertanyaan pengetahuan umum)

        Pesan yang akan dianalisis: {message}'''
    )

    is_waterlevel_question_prompt = PromptTemplate(
        input_variables=["message"],
        template='''Jawab HANYA dengan 'True' atau 'False' tanpa tambahan kata lain.
        Anda diminta untuk memutuskan apakah pesan yang diberikan merupakan pertanyaan 
        yang menanyakan ketinggian muka air pada suatu saluran irigasi. 
        Jawab 'True' jika pesan tersebut adalah pertanyaan yang menanyakan 
        ketinggian muka air pada saluran irigasi (baik menggunakan tanda tanya maupun tidak), dan 'False' jika bukan.
        
        Pesan asli: {message}'''
    )

    summarize_prompt = PromptTemplate(
        input_variables=["message"],
        template='''Tugas buatlah ringkasan dari teks yang diberikan.
        langsung tulis ringkasan saja, tanpa tambahan kalimat apapun,
        abaikan jika ada bagian dari teks yang menurutmu kurang atau tidak relevan.
        teks: {message}'''
    )

@dataclass
class responses:
    identity_question_response = [
        "Saya adalah asisten virtual WRKMC, sebuah sistem kecerdasan buatan yang dirancang untuk membantu Anda dengan berbagai pertanyaan dan tugas.",
        "Perkenalkan, saya asisten AI WRKMC. Saya di sini untuk membantu Anda dengan informasi dan tugas-tugas yang Anda perlukan.",
        "Halo! Saya adalah AI asisten dari WRKMC. Saya siap membantu Anda dengan berbagai pertanyaan dan tugas sesuai kemampuan saya.",
        "Saya adalah sistem AI yang dikembangkan oleh WRKMC untuk membantu pengguna seperti Anda. Senang bertemu dengan Anda!",
        "Nama saya adalah asisten WRKMC, sebuah AI yang dirancang untuk memberikan bantuan dan informasi kepada Anda."
    ]

class Config:
    """Main configuration class."""
    
    def __init__(self):
        self.paths = PathConfig()
        self.splitter = SplitterConfig()
        self.water_level = WaterLevelConfig()
        self.documenProcessing = DocumentProcessingConfig()
        self.promptTemplate = promptTemplate()
        self.response = responses()
        
        # Embedding models
        self._init_embedding_models()
        
        # LLM models
        self._init_llm_models()

    def _init_embedding_models(self) -> None:
        self.embedding_model_1 = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001"
        )
        self.embedding_model_2 = HuggingFaceEmbeddings(
            model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )
        self.EMBEDDING_MODEL = self.embedding_model_2

    def _init_llm_models(self) -> None:
        self.llm_configs = {
            'model_1': LLMConfig(
                name='gemini-1.5-flash',
                temperature=0.5,
                model=ChatGoogleGenerativeAI(
                    model='gemini-1.5-flash',
                    temperature=0.5
                )
            ),
            # 'model_2': LLMConfig(
            #     name='google/gemini-pro-1.5-exp',
            #     temperature=0.0,
            #     model=OpenRouterLLM(
            #         api_key=os.getenv("OPENROUTER_APIKEY"),
            #         model_name='google/gemini-pro-1.5-exp',
            #         temperature=0.0
            #     )
            # ),
            'model_3': LLMConfig(
                name='gemini-1.5-flash',
                temperature=0.0,
                model=ChatGoogleGenerativeAI(
                    model='gemini-1.5-flash',
                    temperature=0.0
                )
            )
        }

    def location_config(self) -> None:
        self.location = [
            {"name": "sapon", "path": self.paths.sapon, "url": self.water_level.sapon_url},
            {"name": "kedungputri", "path": self.paths.kedungputri, "url": self.water_level.kedungputri_url}
        ]

        return self.location

# Create global config instance
config = Config()