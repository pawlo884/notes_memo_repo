from typing import Any
import streamlit as st
from streamlit.errors import StreamlitSecretNotFoundError
from audiorecorder import audiorecorder  # type: ignore
from io import BytesIO
from dotenv import dotenv_values
from openai import OpenAI
from hashlib import md5
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue, Range
from datetime import datetime
import tempfile
import os
import boto3
from botocore.client import Config
import uuid
import psycopg2
from psycopg2.extras import RealDictCursor
import json
import yt_dlp
import re
import traceback
import sys
import pickle
import hashlib
import time
import pytesseract
from PIL import Image


# zmienne globalne

env = dotenv_values(".env")

# Ścieżka do pliku z trwałymi sesjami
SESSION_FILE = ".streamlit_session.pkl"


def log_error(error, context=""):
    """Loguje błąd z kontekstem i pełnym traceback"""
    error_msg = f"🚨 BŁĄD: {str(error)}"
    if context:
        error_msg = f"🚨 BŁĄD w {context}: {str(error)}"

    # Wyświetl w Streamlit
    st.error(error_msg)

    # Pokaż szczegóły w expanderze
    with st.expander("🔍 Szczegóły błędu (kliknij aby rozwinąć)"):
        st.code(traceback.format_exc(), language="python")

        # Dodatkowe informacje
        st.write("**Typ błędu:**", type(error).__name__)
        st.write("**Plik:**", sys.exc_info()[2].tb_frame.f_code.co_filename)
        st.write("**Linia:**", sys.exc_info()[2].tb_lineno)

    # Zapisz do logów (w trybie deweloperskim)
    if st.session_state.get("debug_mode", False):
        print(f"DEBUG ERROR: {error_msg}")
        print(traceback.format_exc())


def show_error_toast(error_msg):
    """Pokazuje błąd jako toast (szybko znika)"""
    st.toast(f"⚠️ {error_msg}", icon="⚠️")


def save_persistent_session(username, remember_me=True):
    """Zapisuje trwałą sesję do pliku"""
    try:
        session_data = {
            "username": username,
            "login_time": time.time(),
            "remember_me": remember_me,
            "session_id": hashlib.md5(f"{username}{time.time()}".encode()).hexdigest()[:16]
        }
        
        with open(SESSION_FILE, 'wb') as f:
            pickle.dump(session_data, f)
        return True
    except Exception as e:
        log_error(e, "zapisywanie trwałej sesji")
        return False


def load_persistent_session():
    """Ładuje trwałą sesję z pliku"""
    try:
        if not os.path.exists(SESSION_FILE):
            return None
            
        with open(SESSION_FILE, 'rb') as f:
            session_data = pickle.load(f)
            
        # Sprawdź czy sesja nie wygasła (30 dni)
        current_time = time.time()
        session_duration = 30 * 24 * 60 * 60  # 30 dni w sekundach
        
        if current_time - session_data.get("login_time", 0) < session_duration:
            return session_data
        else:
            # Sesja wygasła, usuń plik
            delete_persistent_session()
            return None
            
    except Exception as e:
        log_error(e, "ładowanie trwałej sesji")
        return None


def delete_persistent_session():
    """Usuwa plik z trwałą sesją"""
    try:
        if os.path.exists(SESSION_FILE):
            os.remove(SESSION_FILE)
        return True
    except Exception as e:
        log_error(e, "usuwanie trwałej sesji")
        return False


try:
    if "OPENAI_API_KEY" in st.secrets:
        env["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    if "QDRANT_URL" in st.secrets:
        env["QDRANT_URL"] = st.secrets["QDRANT_URL"]
    if "QDRANT_API_KEY" in st.secrets:
        env["QDRANT_API_KEY"] = st.secrets["QDRANT_API_KEY"]
    if "DO_SPACES_KEY" in st.secrets:
        env["DO_SPACES_KEY"] = st.secrets["DO_SPACES_KEY"]
    if "DO_SPACES_SECRET" in st.secrets:
        env["DO_SPACES_SECRET"] = st.secrets["DO_SPACES_SECRET"]
    if "DO_SPACES_REGION" in st.secrets:
        env["DO_SPACES_REGION"] = st.secrets["DO_SPACES_REGION"]
    if "DO_SPACES_BUCKET" in st.secrets:
        env["DO_SPACES_BUCKET"] = st.secrets["DO_SPACES_BUCKET"]
    if "APP_USERNAME" in st.secrets:
        env["APP_USERNAME"] = st.secrets["APP_USERNAME"]
    if "APP_PASSWORD" in st.secrets:
        env["APP_PASSWORD"] = st.secrets["APP_PASSWORD"]
    if "POSTGRES_HOST" in st.secrets:
        env["POSTGRES_HOST"] = st.secrets["POSTGRES_HOST"]
    if "POSTGRES_PORT" in st.secrets:
        env["POSTGRES_PORT"] = st.secrets["POSTGRES_PORT"]
    if "POSTGRES_DB" in st.secrets:
        env["POSTGRES_DB"] = st.secrets["POSTGRES_DB"]
    if "POSTGRES_USER" in st.secrets:
        env["POSTGRES_USER"] = st.secrets["POSTGRES_USER"]
    if "POSTGRES_PASSWORD" in st.secrets:
        env["POSTGRES_PASSWORD"] = st.secrets["POSTGRES_PASSWORD"]
    if "POSTGRES_SSLMODE" in st.secrets:
        env["POSTGRES_SSLMODE"] = st.secrets["POSTGRES_SSLMODE"]
except StreamlitSecretNotFoundError:
    # secrets.toml nie istnieje, używamy tylko .env
    pass

AUDIO_TRANSCRIPTION_MODEL = "whisper-1"

EMBEDDING_MODEL = "text-embedding-3-large"

EMBEDDING_DIM = 1536

QDRANT_COLLECTION_NAME = "app_note_v1"

title = "Audio & Video Notes"

# koniec zmiennych globalnych

st.set_page_config(page_title=title,
                   page_icon=":microphone:", layout="centered")

st.title(title)

# Tryb debugowania (w prawym górnym rogu)
col1, col2 = st.columns([4, 1])
with col1:
    pass
with col2:
    debug_mode = st.checkbox(
        "🐛 Debug", help="Włącza szczegółowe logowanie błędów")
    if debug_mode:
        st.session_state["debug_mode"] = True
    else:
        st.session_state["debug_mode"] = False

add_tab, search_tab, browse_tab, manage_categories_tab = st.tabs(
    ["Dodaj notatkę", "Szukaj notatki", "Przeglądaj notatki", "Zarządzaj kategoriami"])

# openai_client = get_openai_client()


def get_openai_client():
    return OpenAI(api_key=st.session_state["openai_api_key"])


@st.cache_resource
def get_spaces_client():
    """Tworzy klienta boto3 dla DigitalOcean Spaces"""
    required_keys = ["DO_SPACES_KEY", "DO_SPACES_SECRET",
                     "DO_SPACES_REGION", "DO_SPACES_BUCKET"]

    # Sprawdź czy wszystkie klucze istnieją
    missing_keys = [key for key in required_keys if key not in env]
    if missing_keys:
        return None

    # Sprawdź czy klucze nie są puste
    empty_keys = [key for key in required_keys if not str(
        env.get(key, "")).strip()]
    if empty_keys:
        return None

    try:
        session = boto3.session.Session()
        client = session.client(
            's3',
            region_name=env["DO_SPACES_REGION"],
            endpoint_url=f'https://{env["DO_SPACES_REGION"]}.digitaloceanspaces.com',
            aws_access_key_id=env["DO_SPACES_KEY"],
            aws_secret_access_key=env["DO_SPACES_SECRET"],
            config=Config(signature_version='s3v4')
        )
        return client
    except Exception as e:
        st.warning(f"⚠️ DigitalOcean Spaces niedostępne: {str(e)}")
        return None


def upload_file_to_spaces(file_bytes, file_extension, content_type):
    """Uploaduje plik do DigitalOcean Spaces i zwraca URL"""
    spaces_client = get_spaces_client()

    if not spaces_client or "DO_SPACES_BUCKET" not in env:
        return None

    # Generuj unikalną nazwę pliku
    file_id = str(uuid.uuid4())
    filename = f"app_note_v1/{file_id}.{file_extension}"

    try:
        # Upload do Spaces
        spaces_client.put_object(
            Bucket=env["DO_SPACES_BUCKET"],
            Key=filename,
            Body=file_bytes,
            ACL='public-read',
            ContentType=content_type,
            Metadata={
                'uploaded-from': 'audio-video-notes-app'
            }
        )

        # Zwróć publiczny URL
        url = f'https://{env["DO_SPACES_BUCKET"]}.{env["DO_SPACES_REGION"]}.digitaloceanspaces.com/{filename}'
        return url
    except Exception as e:
        log_error(e, "upload do Spaces")
        return None


def get_file_from_spaces_url(url):
    """Pobiera plik z Spaces po URL (jeśli potrzebne do odtwarzania)"""
    spaces_client = get_spaces_client()

    if not spaces_client or not url:
        return None

    try:
        # Wyciągnij nazwę pliku z URL
        filename = url.split('.digitaloceanspaces.com/')[-1]

        response = spaces_client.get_object(
            Bucket=env["DO_SPACES_BUCKET"],
            Key=filename
        )
        return response['Body'].read()
    except Exception as e:
        log_error(e, "pobieranie z Spaces")
        return None


def delete_file_from_spaces(url: str) -> bool:
    """Usuwa plik w Spaces na podstawie pełnego URL. Zwraca True, jeśli sukces."""
    spaces_client = get_spaces_client()

    if not spaces_client or not url:
        return False

    try:
        # Key to część ścieżki po domenie *.digitaloceanspaces.com/
        key = url.split('.digitaloceanspaces.com/')[-1]

        spaces_client.delete_object(
            Bucket=env["DO_SPACES_BUCKET"],
            Key=key
        )
        return True
    except Exception as e:
        log_error(e, "usuwanie pliku ze Spaces")
        return False


def transcribe_audio(audio_bytes, include_timestamps=False):
    """
    Transkrybuje audio, opcjonalnie zwracając timestampy

    Returns:
        - Jeśli include_timestamps=False: str (tylko tekst)
        - Jeśli include_timestamps=True: dict z 'text' i 'segments'
    """
    openai_client = get_openai_client()
    audio_file = BytesIO(audio_bytes)
    audio_file.name = "audio.mp3"
    transcript = openai_client.audio.transcriptions.create(
        file=audio_file,
        model=AUDIO_TRANSCRIPTION_MODEL,
        response_format="verbose_json",
        timestamp_granularities=["segment"]  # Timestampy dla segmentów
    )

    if include_timestamps:
        # Zwróć pełną transkrypcję z timestampami
        return {
            "text": transcript.text,
            "segments": [
                {
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "text": getattr(seg, "text", "")
                }
                for seg in (transcript.segments or [])
            ]
        }

    # Zwróć tylko tekst (backward compatibility)
    return transcript.text


def extract_audio_from_video(video_bytes):
    """Ekstraktuje audio z pliku wideo"""
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(video_bytes)
        temp_video_path = temp_video.name

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as temp_audio:
        temp_audio_path = temp_audio.name

    try:
        # Wczytaj wideo i wyeksportuj audio
        from moviepy.editor import VideoFileClip
        video = VideoFileClip(temp_video_path)
        video.audio.write_audiofile(temp_audio_path, logger=None)
        video.close()

        # Wczytaj audio jako bytes
        with open(temp_audio_path, "rb") as f:
            audio_bytes = f.read()

        return audio_bytes
    finally:
        # Usuń tymczasowe pliki
        if os.path.exists(temp_video_path):
            os.unlink(temp_video_path)
        if os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


def download_instagram_video(url):
    """
    Pobiera wideo z Instagram (Reels, posty, IGTV)

    Returns:
        dict: {
            'video_bytes': bytes,
            'title': str,
            'author': str,
            'description': str,
            'url': str
        }
    """

    # Konfiguracja yt-dlp
    ydl_opts = {
        'format': 'best',
        'quiet': True,
        'no_warnings': True,
        'extract_flat': False,
        'outtmpl': '%(id)s.%(ext)s',
    }

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            ydl_opts['outtmpl'] = os.path.join(temp_dir, '%(id)s.%(ext)s')

            # Pobierz informacje i wideo
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)

                # Znajdź pobrany plik
                video_file = None
                for file in os.listdir(temp_dir):
                    if file.startswith(info.get('id', '')):
                        video_file = os.path.join(temp_dir, file)
                        break

                if not video_file or not os.path.exists(video_file):
                    raise Exception("Nie udało się pobrać wideo")

                # Wczytaj wideo jako bytes
                with open(video_file, 'rb') as f:
                    video_bytes = f.read()

                # Wyciągnij metadane
                return {
                    'video_bytes': video_bytes,
                    'title': info.get('title', 'Instagram Video'),
                    'author': info.get('uploader', 'Nieznany'),
                    'description': info.get('description', ''),
                    'url': url,
                    'duration': info.get('duration', 0),
                    'timestamp': info.get('timestamp', None)
                }

    except Exception as e:
        raise Exception(f"Błąd pobierania z Instagram: {str(e)}")


def is_instagram_url(url):
    """Sprawdza czy URL jest z Instagram"""
    instagram_patterns = [
        r'instagram\.com/p/',        # Posty
        r'instagram\.com/reel/',     # Reels
        r'instagram\.com/tv/',       # IGTV
        r'instagram\.com/stories/',  # Stories (może nie działać)
    ]

    return any(re.search(pattern, url) for pattern in instagram_patterns)


def extract_text_from_image(image_bytes):
    """
    Ekstraktuje tekst z obrazka używając OCR (Tesseract)
    
    Args:
        image_bytes: bytes - dane obrazka
        
    Returns:
        str - wyekstraktowany tekst
    """
    try:
        # Otwórz obrazek z bytes
        image = Image.open(BytesIO(image_bytes))
        
        # Konwertuj na RGB jeśli potrzeba
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Użyj Tesseract do OCR
        # Ustaw język na polski + angielski
        text = pytesseract.image_to_string(image, lang='pol+eng')
        
        return text.strip()
    except Exception as e:
        raise Exception(f"Błąd podczas ekstrakcji tekstu z obrazka: {str(e)}")


def process_multiple_images(image_files):
    """
    Przetwarza wiele obrazków jednocześnie
    
    Args:
        image_files: lista plików obrazków z Streamlit
        
    Returns:
        dict: {
            'combined_text': str - połączony tekst ze wszystkich obrazków,
            'individual_texts': list - lista tekstów z każdego obrazka,
            'image_data': list - lista danych obrazków (bytes, nazwa, typ)
        }
    """
    combined_text = ""
    individual_texts = []
    image_data = []
    
    for i, uploaded_file in enumerate(image_files):
        try:
            # Wczytaj dane obrazka
            image_bytes = uploaded_file.read()
            file_ext = uploaded_file.name.split('.')[-1].lower()
            content_type = f"image/{file_ext}"
            
            # Zapisz dane obrazka
            image_data.append({
                'bytes': image_bytes,
                'name': uploaded_file.name,
                'extension': file_ext,
                'content_type': content_type
            })
            
            # Ekstraktuj tekst
            with st.spinner(f"Przetwarzam obrazek {i+1}/{len(image_files)}: {uploaded_file.name}"):
                text = extract_text_from_image(image_bytes)
                individual_texts.append({
                    'filename': uploaded_file.name,
                    'text': text
                })
                
                # Dodaj do połączonego tekstu
                if text:
                    combined_text += f"\n\n--- Obrazek {i+1}: {uploaded_file.name} ---\n{text}"
                else:
                    combined_text += f"\n\n--- Obrazek {i+1}: {uploaded_file.name} ---\n[Brak tekstu do odczytania]"
                    
        except Exception as e:
            st.error(f"Błąd przetwarzania obrazka {uploaded_file.name}: {str(e)}")
            individual_texts.append({
                'filename': uploaded_file.name,
                'text': f"[BŁĄD: {str(e)}]"
            })
            combined_text += f"\n\n--- Obrazek {i+1}: {uploaded_file.name} ---\n[BŁĄD: {str(e)}]"
    
    return {
        'combined_text': combined_text.strip(),
        'individual_texts': individual_texts,
        'image_data': image_data
    }


@st.cache_resource
def get_qdrant_client():
    return QdrantClient(
        url=env["QDRANT_URL"],
        api_key=env["QDRANT_API_KEY"],
    )


def assure_db_collection_exists():
    qdrant_client = get_qdrant_client()
    if not qdrant_client.collection_exists(QDRANT_COLLECTION_NAME):
        print("Tworzę kolekcję")
        qdrant_client.create_collection(
            collection_name=QDRANT_COLLECTION_NAME,
            vectors_config=VectorParams(
                size=EMBEDDING_DIM,
                distance=Distance.COSINE,
            ),
        )
    else:
        print("Kolekcja już istnieje")


# PostgreSQL connection
def get_postgres_connection():
    """Tworzy połączenie z PostgreSQL"""
    if not all(key in env for key in ["POSTGRES_HOST", "POSTGRES_DB", "POSTGRES_USER", "POSTGRES_PASSWORD"]):
        return None

    try:
        conn = psycopg2.connect(
            host=env["POSTGRES_HOST"],
            port=env.get("POSTGRES_PORT", "5432"),
            database=env["POSTGRES_DB"],
            user=env["POSTGRES_USER"],
            password=env["POSTGRES_PASSWORD"],
            sslmode=env.get("POSTGRES_SSLMODE", "require")
        )
        return conn
    except Exception as e:
        st.error(f"Błąd połączenia z PostgreSQL: {str(e)}")
        return None


def init_postgres_tables():
    """Inicjalizuje tabele w PostgreSQL jeśli nie istnieją"""
    conn = get_postgres_connection()
    if not conn:
        return False

    try:
        with conn.cursor() as cur:
            # Odczytaj i wykonaj skrypt init_db.sql
            with open("init_db.sql", "r", encoding="utf-8") as f:
                sql = f.read()
                cur.execute(sql)

            # Aktualizuj constraint jeśli istnieje (dodaj 'instagram' i 'image')
            try:
                cur.execute("""
                    ALTER TABLE notes DROP CONSTRAINT IF EXISTS notes_source_type_check;
                    ALTER TABLE notes ADD CONSTRAINT notes_source_type_check 
                    CHECK (source_type IN ('audio', 'video', 'text', 'instagram', 'image'));
                """)
            except Exception:
                # Ignoruj błąd jeśli constraint nie istnieje
                pass

        conn.commit()
        return True
    except Exception as e:
        log_error(e, "inicjalizacja tabel PostgreSQL")
        conn.rollback()
        return False
    finally:
        conn.close()


def get_embedding(text):
    openai_client = get_openai_client()
    result = openai_client.embeddings.create(
        input=[text],
        model=EMBEDDING_MODEL,
        dimensions=EMBEDDING_DIM,
    )
    return result.data[0].embedding


def generate_search_context(query, note_text, max_length=300):
    """
    Generuje kontekst/streszczenie pokazujące dlaczego notatka pasuje do zapytania
    """
    openai_client = get_openai_client()

    try:
        response = openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "Jesteś asystentem, który pomaga znaleźć relevantne fragmenty tekstu. Wypisz 1-2 zdania pokazujące, które fragmenty notatki pasują do zapytania użytkownika. Cytuj konkretne fragmenty."
                },
                {
                    "role": "user",
                    "content": f"Zapytanie użytkownika: '{query}'\n\nNotatka:\n{note_text[:1000]}\n\nWyjaśnij krótko (max 2 zdania), dlaczego ta notatka pasuje do zapytania, cytując konkretny fragment."
                }
            ],
            max_tokens=150,
            temperature=0.3
        )

        return response.choices[0].message.content.strip()
    except Exception as e:
        # Fallback - pokaż fragment tekstu zawierający słowa z zapytania
        query_words = query.lower().split()
        note_lower = note_text.lower()

        # Znajdź pierwsze wystąpienie któregoś ze słów
        for word in query_words:
            if word in note_lower:
                idx = note_lower.index(word)
                start = max(0, idx - 100)
                end = min(len(note_text), idx + 200)
                fragment = note_text[start:end].strip()
                if start > 0:
                    fragment = "..." + fragment
                if end < len(note_text):
                    fragment = fragment + "..."
                return f"Fragment: {fragment}"

        # Jeśli nie znaleziono, zwróć początek notatki
        return note_text[:max_length] + ("..." if len(note_text) > max_length else "")


def normalize_polish_text(text):
    """
    Normalizuje polski tekst usuwając znaki diakrytyczne
    """
    # Używamy ręcznego mapowania zamiast unidecode (który źle obsługuje polskie znaki)
    polish_map = {
        'ą': 'a', 'ć': 'c', 'ę': 'e', 'ł': 'l', 'ń': 'n',
        'ó': 'o', 'ś': 's', 'ź': 'z', 'ż': 'z',
        'Ą': 'A', 'Ć': 'C', 'Ę': 'E', 'Ł': 'L', 'Ń': 'N',
        'Ó': 'O', 'Ś': 'S', 'Ź': 'Z', 'Ż': 'Z'
    }

    normalized = text.lower()
    for polish, basic in polish_map.items():
        normalized = normalized.replace(polish, basic)

    return normalized


def highlight_text(text, query):
    """
    Podświetla słowa z zapytania w tekście używając embeddings do porównania podobieństwa
    """
    if not query:
        return text

    import re
    import numpy as np

    try:
        # Pobierz klienta OpenAI
        openai_client = get_openai_client()

        # Podziel zapytanie na słowa
        query_words = query.lower().split()
        highlighted_text = text

        # Znajdź wszystkie słowa w tekście
        words_in_text = re.findall(r'\b\w+\b', text)

        # Generuj embeddings dla słów z zapytania (tylko raz)
        query_word_embeddings = {}
        for word in query_words:
            if len(word) < 3:  # Ignoruj bardzo krótkie słowa
                continue

            try:
                response = openai_client.embeddings.create(
                    input=word,
                    model="text-embedding-3-small"
                )
                query_word_embeddings[word] = np.array(
                    response.data[0].embedding)
            except Exception as e:
                continue

        # Sprawdź każde słowo z tekstu
        for word_in_text in words_in_text:
            if len(word_in_text) < 3:  # Ignoruj bardzo krótkie słowa
                continue

            # Sprawdź czy słowo jest identyczne (dokładne dopasowanie)
            exact_match = False
            for query_word in query_words:
                if word_in_text.lower() == query_word.lower():
                    exact_match = True
                    break

            if exact_match:
                pattern = re.compile(
                    r'\b' + re.escape(word_in_text) + r'\b', re.IGNORECASE)
                highlighted_text = pattern.sub(
                    lambda m: f'<mark style="background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{m.group()}</mark>',
                    highlighted_text
                )
                continue

            # Sprawdź podobieństwo semantyczne przez embeddings
            best_similarity = 0

            try:
                response = openai_client.embeddings.create(
                    input=word_in_text,
                    model="text-embedding-3-small"
                )
                text_word_embedding = np.array(response.data[0].embedding)

                # Porównaj z każdym słowem z zapytania
                for query_word, query_embedding in query_word_embeddings.items():
                    # Oblicz podobieństwo cosinusowe
                    similarity = np.dot(text_word_embedding, query_embedding) / (
                        np.linalg.norm(text_word_embedding) *
                        np.linalg.norm(query_embedding)
                    )

                    if similarity > best_similarity:
                        best_similarity = similarity

                # Jeśli podobieństwo jest wysokie (powyżej 0.6), podświetl słowo
                if best_similarity > 0.6:
                    pattern = re.compile(
                        r'\b' + re.escape(word_in_text) + r'\b', re.IGNORECASE)
                    highlighted_text = pattern.sub(
                        lambda m: f'<mark style="background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{m.group()}</mark>',
                        highlighted_text
                    )
            except Exception:
                continue

        return highlighted_text

    except Exception:
        # Fallback do prostego dopasowania
        return highlight_text_fallback(text, query)


def highlight_text_fallback(text, query):
    """
    Fallback highlighting używający prostego dopasowania tekstu
    """
    if not query:
        return text

    import re

    query_words = query.lower().split()
    highlighted_text = text

    for word in query_words:
        if len(word) < 3:
            continue

        # Podświetl dokładne dopasowania
        pattern = re.compile(r'\b' + re.escape(word) + r'\b', re.IGNORECASE)
        highlighted_text = pattern.sub(
            lambda m: f'<mark style="background-color: #FFEB3B; padding: 2px 4px; border-radius: 3px; font-weight: bold;">{m.group()}</mark>',
            highlighted_text
        )

    return highlighted_text


def add_note_to_db(note_text, categories=None, source_type="audio", media_url=None, media_type=None, timestamps=None, multiple_images_data=None):
    """Zapisuje notatkę do PostgreSQL i Qdrant

    Args:
        timestamps: Lista segmentów z timestampami {'start': float, 'end': float, 'text': str}
        multiple_images_data: Lista danych obrazków dla source_type='image'
    """
    conn = get_postgres_connection()

    # Jeśli brak PostgreSQL, użyj starej metody (tylko Qdrant)
    if not conn:
        qdrant_client = get_qdrant_client()
        points_count = qdrant_client.count(
            collection_name=QDRANT_COLLECTION_NAME,
            exact=True,
        )
        timestamp = datetime.now().isoformat()
        payload = {
            "text": note_text,
            "timestamp": timestamp,
            "source_type": source_type,
            "categories": categories or [],
            "media_url": media_url,
            "media_type": media_type,
            "timestamps": timestamps,  # Timestampy segmentów
            "multiple_images_data": multiple_images_data,  # Dane o wielu obrazkach
        }
        qdrant_client.upsert(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=points_count.count + 1,
                    vector=get_embedding(text=note_text),
                    payload=payload,
                )
            ]
        )
        return

    # Nowa metoda z PostgreSQL + Qdrant
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # 1. Dodaj notatkę do Qdrant i pobierz ID
            qdrant_client = get_qdrant_client()
            points_count = qdrant_client.count(
                collection_name=QDRANT_COLLECTION_NAME,
                exact=True,
            )
            qdrant_id = points_count.count + 1

            # 2. Zapisz do PostgreSQL
            cur.execute("""
                INSERT INTO notes (text, source_type, media_url, media_type, qdrant_id, timestamps, multiple_images_data)
                VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                RETURNING id
            """, (note_text, source_type, media_url, media_type, qdrant_id, json.dumps(timestamps) if timestamps else None, json.dumps(multiple_images_data) if multiple_images_data else None))

            note_id = cur.fetchone()["id"]

            # 3. Obsługa kategorii
            if categories:
                for cat_name in categories:
                    # Utwórz kategorię jeśli nie istnieje
                    cur.execute("""
                        INSERT INTO categories (name)
                        VALUES (%s)
                        ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                        RETURNING id
                    """, (cat_name,))
                    cat_id = cur.fetchone()["id"]

                    # Powiąż notatkę z kategorią
                    cur.execute("""
                        INSERT INTO note_categories (note_id, category_id)
                        VALUES (%s, %s)
                        ON CONFLICT DO NOTHING
                    """, (note_id, cat_id))

            conn.commit()

            # 4. Dodaj do Qdrant
            timestamp = datetime.now().isoformat()
            payload = {
                "text": note_text,
                "timestamp": timestamp,
                "source_type": source_type,
                "categories": categories or [],
                "media_url": media_url,
                "media_type": media_type,
                "postgres_id": note_id,
                "timestamps": timestamps,  # Timestampy segmentów
                "multiple_images_data": multiple_images_data,  # Dane o wielu obrazkach
            }

            qdrant_client.upsert(
                collection_name=QDRANT_COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=qdrant_id,
                        vector=get_embedding(text=note_text),
                        payload=payload,
                    )
                ]
            )

    except Exception as e:
        conn.rollback()
        st.error(f"Błąd zapisu do bazy: {str(e)}")
    finally:
        conn.close()


def list_notes_from_db(search_query=None, category_filter=None, date_from=None, date_to=None, limit=50) -> list[Any]:
    """Pobiera notatki z PostgreSQL lub Qdrant"""
    conn = get_postgres_connection()

    # Jeśli brak PostgreSQL, użyj starej metody (tylko Qdrant)
    if not conn:
        qdrant_client = get_qdrant_client()
        filter_conditions = []
        if category_filter:
            filter_conditions.append(FieldCondition(
                key="categories", match=MatchValue(value=category_filter)))
        if date_from:
            filter_conditions.append(FieldCondition(
                key="timestamp", range=Range(gte=date_from.isoformat())))
        if date_to:
            date_to_end = datetime.combine(date_to, datetime.max.time())
            filter_conditions.append(FieldCondition(
                key="timestamp", range=Range(lte=date_to_end.isoformat())))
        query_filter = Filter(
            must=filter_conditions) if filter_conditions else None

        if not search_query:
            notes = qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION_NAME, limit=limit, scroll_filter=query_filter)[0]
            result = []
            for record in notes:
                if record.payload:
                    result.append({
                        "text": record.payload.get("text", ""),
                        "score": None,
                        "timestamp": record.payload.get("timestamp", ""),
                        "source_type": record.payload.get("source_type", "audio"),
                        "categories": record.payload.get("categories", []),
                        "media_url": record.payload.get("media_url"),
                        "media_type": record.payload.get("media_type"),
                        # Timestampy segmentów
                        "timestamps": record.payload.get("timestamps"),
                        # Dane o obrazkach
                        "multiple_images_data": record.payload.get("multiple_images_data"),
                    })
            result.sort(key=lambda x: x["timestamp"], reverse=True)
            return result
        else:
            notes = qdrant_client.query_points(collection_name=QDRANT_COLLECTION_NAME, query=get_embedding(
                text=search_query), limit=limit, query_filter=query_filter).points
            result = []
            for record in notes:
                if record.payload:
                    result.append({
                        "text": record.payload.get("text", ""),
                        "score": record.score,
                        "timestamp": record.payload.get("timestamp", ""),
                        "source_type": record.payload.get("source_type", "audio"),
                        "categories": record.payload.get("categories", []),
                        "media_url": record.payload.get("media_url"),
                        "media_type": record.payload.get("media_type"),
                        # Timestampy segmentów
                        "timestamps": record.payload.get("timestamps"),
                        # Dane o obrazkach
                        "multiple_images_data": record.payload.get("multiple_images_data"),
                    })
            return result

    # Nowa metoda z PostgreSQL
    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Bez wyszukiwania semantycznego - pobierz z PostgreSQL
            if not search_query:
                sql = """
                    SELECT 
                        n.id,
                        n.text,
                        n.timestamp,
                        n.source_type,
                        n.media_url,
                        n.media_type,
                        n.qdrant_id,
                        n.timestamps,
                        n.multiple_images_data,
                        COALESCE(
                            json_agg(c.name) FILTER (WHERE c.name IS NOT NULL),
                            '[]'::json
                        ) as categories
                    FROM notes n
                    LEFT JOIN note_categories nc ON n.id = nc.note_id
                    LEFT JOIN categories c ON nc.category_id = c.id
                    WHERE 1=1
                """
                params = []

                if category_filter:
                    sql += " AND c.name = %s"
                    params.append(category_filter)

                if date_from:
                    sql += " AND n.timestamp >= %s"
                    params.append(date_from)

                if date_to:
                    date_to_end = datetime.combine(
                        date_to, datetime.max.time())
                    sql += " AND n.timestamp <= %s"
                    params.append(date_to_end)

                sql += " GROUP BY n.id ORDER BY n.timestamp DESC LIMIT %s"
                params.append(limit)

                cur.execute(sql, params)
                rows = cur.fetchall()

                result = []
                for row in rows:
                    result.append({
                        "id": row["id"],
                        "qdrant_id": row["qdrant_id"],
                        "text": row["text"],
                        "score": None,
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else "",
                        "source_type": row["source_type"],
                        "categories": row["categories"] if isinstance(row["categories"], list) else json.loads(row["categories"]),
                        "media_url": row["media_url"],
                        "media_type": row["media_type"],
                        # JSONB z timestampami
                        "timestamps": row["timestamps"],
                        # Dane o obrazkach
                        "multiple_images_data": row.get("multiple_images_data"),
                    })
                return result

            # Z wyszukiwaniem semantycznym - użyj Qdrant + PostgreSQL
            else:
                qdrant_client = get_qdrant_client()

                # Wyszukaj w Qdrant
                notes = qdrant_client.query_points(
                    collection_name=QDRANT_COLLECTION_NAME,
                    query=get_embedding(text=search_query),
                    limit=limit * 2,  # Pobierz więcej, bo będziemy filtrować
                ).points

                # Pobierz IDs z PostgreSQL
                postgres_ids = [record.payload.get(
                    "postgres_id") for record in notes if record.payload.get("postgres_id")]

                if not postgres_ids:
                    return []

                # Pobierz szczegóły z PostgreSQL
                sql = """
                    SELECT 
                        n.id,
                        n.text,
                        n.timestamp,
                        n.source_type,
                        n.media_url,
                        n.media_type,
                        n.qdrant_id,
                        n.timestamps,
                        n.multiple_images_data,
                        COALESCE(
                            json_agg(c.name) FILTER (WHERE c.name IS NOT NULL),
                            '[]'::json
                        ) as categories
                    FROM notes n
                    LEFT JOIN note_categories nc ON n.id = nc.note_id
                    LEFT JOIN categories c ON nc.category_id = c.id
                    WHERE n.id = ANY(%s)
                """
                params = [postgres_ids]

                if category_filter:
                    sql += " AND c.name = %s"
                    params.append(category_filter)

                if date_from:
                    sql += " AND n.timestamp >= %s"
                    params.append(date_from)

                if date_to:
                    date_to_end = datetime.combine(
                        date_to, datetime.max.time())
                    sql += " AND n.timestamp <= %s"
                    params.append(date_to_end)

                sql += " GROUP BY n.id ORDER BY n.timestamp DESC LIMIT %s"
                params.append(limit)

                cur.execute(sql, params)
                rows = cur.fetchall()

                # Mapuj score z Qdrant
                score_map = {record.payload.get(
                    "postgres_id"): record.score for record in notes if record.payload.get("postgres_id")}

                result = []
                for row in rows:
                    result.append({
                        "id": row["id"],
                        "qdrant_id": row["qdrant_id"],
                        "text": row["text"],
                        "score": score_map.get(row["id"]),
                        "timestamp": row["timestamp"].isoformat() if row["timestamp"] else "",
                        "source_type": row["source_type"],
                        "categories": row["categories"] if isinstance(row["categories"], list) else json.loads(row["categories"]),
                        "media_url": row["media_url"],
                        "media_type": row["media_type"],
                        # JSONB z timestampami
                        "timestamps": row["timestamps"],
                        # Dane o obrazkach
                        "multiple_images_data": row.get("multiple_images_data"),
                    })

                # Sortuj wg score
                result.sort(key=lambda x: x["score"]
                            if x["score"] else 0, reverse=True)
                return result

    except Exception as e:
        st.error(f"Błąd pobierania notatek: {str(e)}")
        return []
    finally:
        conn.close()


def get_all_categories() -> list[str]:
    """Pobiera wszystkie unikalne kategorie z bazy"""
    conn = get_postgres_connection()

    # Jeśli brak PostgreSQL, użyj starej metody
    if not conn:
        qdrant_client = get_qdrant_client()
        notes = qdrant_client.scroll(
            collection_name=QDRANT_COLLECTION_NAME,
            limit=1000,
            with_payload=True,
        )[0]
        categories = set()
        for record in notes:
            if record.payload and "categories" in record.payload:
                categories.update(record.payload["categories"])
        return sorted(list(categories))

    # Nowa metoda z PostgreSQL
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT name FROM categories ORDER BY name")
            rows = cur.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        st.error(f"Błąd pobierania kategorii: {str(e)}")
        return []
    finally:
        conn.close()


def delete_note(note_id, qdrant_id):
    """Usuwa notatkę z PostgreSQL, Qdrant oraz plik z DigitalOcean Spaces (jeśli istnieje)."""
    conn = get_postgres_connection()

    # Jeśli brak PostgreSQL, usuń tylko z Qdrant
    if not conn:
        qdrant_client = get_qdrant_client()
        try:
            qdrant_client.delete(
                collection_name=QDRANT_COLLECTION_NAME,
                points_selector=[qdrant_id]
            )
            return True
        except Exception as e:
            st.error(f"Błąd usuwania z Qdrant: {str(e)}")
            return False

    # Usuń z PostgreSQL (zwróć media_url, jeśli istnieje) i Qdrant
    try:
        media_url = None
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Pobierz URL pliku przed usunięciem rekordu
            try:
                cur.execute(
                    "SELECT media_url FROM notes WHERE id = %s", (note_id,))
                row = cur.fetchone()
                media_url = (row or {}).get("media_url")
            except Exception:
                media_url = None

            cur.execute("DELETE FROM notes WHERE id = %s", (note_id,))
        conn.commit()

        # Usuń z Qdrant
        qdrant_client = get_qdrant_client()
        qdrant_client.delete(
            collection_name=QDRANT_COLLECTION_NAME,
            points_selector=[qdrant_id]
        )

        # Usuń plik ze Spaces, jeśli mamy URL
        if media_url:
            delete_file_from_spaces(media_url)

        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Błąd usuwania notatki: {str(e)}")
        return False
    finally:
        conn.close()


def update_note_categories(note_id, new_categories):
    """Aktualizuje kategorie notatki w PostgreSQL i Qdrant"""
    conn = get_postgres_connection()

    if not conn:
        st.error("Wymaga połączenia z PostgreSQL")
        return False

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Pobierz qdrant_id
            cur.execute(
                "SELECT qdrant_id FROM notes WHERE id = %s", (note_id,))
            result = cur.fetchone()
            if not result:
                st.error("Notatka nie znaleziona")
                return False

            qdrant_id = result["qdrant_id"]

            # Usuń stare powiązania kategorii
            cur.execute(
                "DELETE FROM note_categories WHERE note_id = %s", (note_id,))

            # Dodaj nowe kategorie
            for cat_name in new_categories:
                # Utwórz kategorię jeśli nie istnieje
                cur.execute("""
                    INSERT INTO categories (name)
                    VALUES (%s)
                    ON CONFLICT (name) DO UPDATE SET name = EXCLUDED.name
                    RETURNING id
                """, (cat_name,))
                cat_id = cur.fetchone()["id"]

                # Powiąż notatkę z kategorią
                cur.execute("""
                    INSERT INTO note_categories (note_id, category_id)
                    VALUES (%s, %s)
                    ON CONFLICT DO NOTHING
                """, (note_id, cat_id))

        conn.commit()

        # Aktualizuj w Qdrant
        qdrant_client = get_qdrant_client()
        qdrant_client.set_payload(
            collection_name=QDRANT_COLLECTION_NAME,
            payload={"categories": new_categories},
            points=[qdrant_id]
        )

        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Błąd aktualizacji kategorii: {str(e)}")
        return False
    finally:
        conn.close()


def update_note_text(note_id, new_text):
    """Aktualizuje treść notatki w PostgreSQL i Qdrant"""
    conn = get_postgres_connection()

    if not conn:
        st.error("Wymaga połączenia z PostgreSQL")
        return False

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            # Pobierz qdrant_id
            cur.execute(
                "SELECT qdrant_id FROM notes WHERE id = %s", (note_id,))
            result = cur.fetchone()
            if not result:
                st.error("Notatka nie znaleziona")
                return False

            qdrant_id = result["qdrant_id"]

            # Aktualizuj tekst w PostgreSQL
            cur.execute("""
                UPDATE notes 
                SET text = %s, updated_at = CURRENT_TIMESTAMP 
                WHERE id = %s
            """, (new_text, note_id))

        conn.commit()

        # Aktualizuj w Qdrant (tekst i embedding)
        qdrant_client = get_qdrant_client()
        new_embedding = get_embedding(new_text)

        # Aktualizuj zarówno wektor jak i payload
        qdrant_client.update_vectors(
            collection_name=QDRANT_COLLECTION_NAME,
            points=[
                PointStruct(
                    id=qdrant_id,
                    vector=new_embedding
                )
            ]
        )

        qdrant_client.set_payload(
            collection_name=QDRANT_COLLECTION_NAME,
            payload={"text": new_text},
            points=[qdrant_id]
        )

        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Błąd aktualizacji treści: {str(e)}")
        return False
    finally:
        conn.close()


def get_category_stats():
    """Pobiera statystyki użycia kategorii"""
    conn = get_postgres_connection()

    if not conn:
        return []

    try:
        with conn.cursor(cursor_factory=RealDictCursor) as cur:
            cur.execute("""
                SELECT 
                    c.id,
                    c.name,
                    COUNT(nc.note_id) as note_count
                FROM categories c
                LEFT JOIN note_categories nc ON c.id = nc.category_id
                GROUP BY c.id, c.name
                ORDER BY note_count DESC, c.name
            """)
            return cur.fetchall()
    except Exception as e:
        st.error(f"Błąd pobierania statystyk: {str(e)}")
        return []
    finally:
        conn.close()


def delete_category(category_id):
    """Usuwa kategorię (CASCADE usuwa też powiązania z notatkami)"""
    conn = get_postgres_connection()

    if not conn:
        st.error("Wymaga połączenia z PostgreSQL")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM categories WHERE id = %s", (category_id,))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Błąd usuwania kategorii: {str(e)}")
        return False
    finally:
        conn.close()


def rename_category(category_id, new_name):
    """Zmienia nazwę kategorii"""
    conn = get_postgres_connection()

    if not conn:
        st.error("Wymaga połączenia z PostgreSQL")
        return False

    try:
        with conn.cursor() as cur:
            cur.execute("""
                UPDATE categories 
                SET name = %s 
                WHERE id = %s
            """, (new_name, category_id))
        conn.commit()
        return True
    except Exception as e:
        conn.rollback()
        st.error(f"Błąd zmiany nazwy kategorii: {str(e)}")
        return False
    finally:
        conn.close()


# MAIN

# System logowania z funkcjonalnością "Remember Me"
def check_password():
    """Sprawdza czy użytkownik jest zalogowany"""

    # Sprawdź czy w .env jest login i hasło
    if "APP_USERNAME" not in env or "APP_PASSWORD" not in env:
        # Brak konfiguracji - aplikacja bez logowania
        return True

    def login_form():
        """Wyświetla formularz logowania"""
        st.title("🔐 Logowanie")
        st.markdown("---")

        with st.form("login_form"):
            username = st.text_input("Login")
            password = st.text_input("Hasło", type="password")
            remember_me = st.checkbox("Zapamiętaj mnie", help="Zostaniesz zalogowany automatycznie przy następnej wizycie")
            submit = st.form_submit_button("Zaloguj się", type="primary")

            if submit:
                if username == env["APP_USERNAME"] and password == env["APP_PASSWORD"]:
                    st.session_state["authenticated"] = True
                    st.session_state["username"] = username
                    
                    # Jeśli użytkownik wybrał "Zapamiętaj mnie", zapisz trwałą sesję
                    if remember_me:
                        save_persistent_session(username, remember_me=True)
                        st.session_state["remember_me"] = True
                    
                    st.success("✅ Zalogowano pomyślnie!")
                    if remember_me:
                        st.info("💾 Zostaniesz zalogowany automatycznie przy następnej wizycie")
                    st.rerun()
                else:
                    st.error("❌ Nieprawidłowy login lub hasło")

    # Sprawdź czy użytkownik ma trwałą sesję (Remember Me)
    if not st.session_state.get("authenticated", False):
        # Sprawdź czy istnieje trwała sesja w pliku
        persistent_session = load_persistent_session()
        if persistent_session and persistent_session.get("remember_me"):
            # Automatyczne logowanie
            st.session_state["authenticated"] = True
            st.session_state["username"] = persistent_session.get("username")
            st.session_state["remember_me"] = True
            st.info("🔄 Automatyczne logowanie z zapamiętanej sesji")

    # Sprawdź czy użytkownik jest zalogowany
    if not st.session_state.get("authenticated", False):
        login_form()
        st.stop()

    # Przycisk wylogowania w sidebar
    with st.sidebar:
        st.write(f"👤 Zalogowany jako: **{st.session_state.get('username')}**")
        
        # Pokaż informację o trwałej sesji
        if st.session_state.get("remember_me"):
            st.caption("💾 Trwała sesja aktywna")
        
        if st.button("🚪 Wyloguj się"):
            # Usuń wszystkie dane sesji
            st.session_state["authenticated"] = False
            st.session_state["username"] = None
            st.session_state["remember_me"] = False
            # Usuń trwałą sesję z pliku
            delete_persistent_session()
            st.rerun()

    return True


# Sprawdź logowanie
check_password()

if not st.session_state.get("openai_api_key"):
    if "OPENAI_API_KEY" in env:
        st.session_state["openai_api_key"] = env["OPENAI_API_KEY"]

    else:
        st.info("Podaj API key OpenAI")
        st.session_state["openai_api_key"] = st.text_input(
            "API key OpenAI", type="password")
        if st.session_state["openai_api_key"]:
            st.rerun()

if not st.session_state.get("openai_api_key"):
    st.stop()
# Session state initialization
if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_bytes_md5"] = None

if "note_audio_bytes" not in st.session_state:
    st.session_state["note_audio_bytes"] = None

if "note_text" not in st.session_state:
    st.session_state["note_text"] = ""

if "note_audio_text" not in st.session_state:
    st.session_state["note_audio_text"] = ""

if "note_categories" not in st.session_state:
    st.session_state["note_categories"] = []

if "source_type" not in st.session_state:
    st.session_state["source_type"] = "audio"

assure_db_collection_exists()

# Inicjalizuj PostgreSQL jeśli jest skonfigurowany
if get_postgres_connection():
    init_postgres_tables()

with add_tab:
    st.subheader("Wybierz źródło notatki")

    source_option = st.radio(
        "Źródło",
        ["🎤 Nagraj audio", "📁 Upload audio/wideo",
            "📱 Pobierz z Instagram", "🖼️ Upload obrazków", "📝 Napisz tekst"],
        horizontal=True
    )

    st.divider()

    if source_option == "🎤 Nagraj audio":
        st.session_state["source_type"] = "audio"
        note_audio = audiorecorder(
            start_prompt="🔴 Nagraj notatkę",
            stop_prompt="⏹️ Zatrzymaj nagrywanie",)

        if note_audio:
            audio = BytesIO()
            note_audio.export(audio, format="mp3")
            st.session_state["note_audio_bytes"] = audio.getvalue()
            current_md5 = md5(st.session_state["note_audio_bytes"]).hexdigest()
            if st.session_state["note_audio_bytes_md5"] != current_md5:
                st.session_state["note_audio_text"] = ""
                st.session_state["note_text"] = ""
                st.session_state["note_audio_bytes_md5"] = current_md5
                st.session_state["media_file_bytes"] = st.session_state["note_audio_bytes"]
                st.session_state["media_file_extension"] = "mp3"
                st.session_state["media_content_type"] = "audio/mp3"

            # Wyświetl informację o długości nagrania
            try:
                from pydub import AudioSegment
                audio_segment = AudioSegment.from_mp3(
                    BytesIO(st.session_state["note_audio_bytes"]))
                duration_seconds = len(audio_segment) / 1000.0
                minutes = int(duration_seconds // 60)
                seconds = int(duration_seconds % 60)
                st.info(f"⏱️ Długość nagrania: {minutes}:{seconds:02d}")
            except Exception:
                pass

            st.audio(st.session_state["note_audio_bytes"], format="audio/mp3")

            if st.button("Transkrypcja", key="transcribe_audio"):
                with st.spinner("Transkrybuję audio..."):
                    st.session_state["note_audio_text"] = transcribe_audio(
                        st.session_state["note_audio_bytes"])

            if st.session_state["note_audio_text"]:
                st.session_state["note_text"] = st.text_area(
                    "Edytuj notatkę", value=st.session_state["note_audio_text"])

    elif source_option == "📁 Upload audio/wideo":
        uploaded_file = st.file_uploader(
            "Wybierz plik audio lub wideo",
            type=["mp3", "wav", "m4a", "ogg", "flac",
                  "mp4", "mov", "avi", "mkv", "webm"],
            help="Audio: MP3, WAV, M4A, OGG, FLAC | Wideo: MP4, MOV, AVI, MKV, WEBM"
        )

        if uploaded_file is not None:
            file_bytes = uploaded_file.read()
            file_ext = uploaded_file.name.split('.')[-1].lower()

            # Określ czy to plik audio czy wideo
            audio_formats = ["mp3", "wav", "m4a", "ogg", "flac"]
            video_formats = ["mp4", "mov", "avi", "mkv", "webm"]

            is_audio = file_ext in audio_formats
            is_video = file_ext in video_formats

            if is_audio:
                st.session_state["source_type"] = "audio"
                st.session_state["media_content_type"] = f"audio/{file_ext if file_ext != 'm4a' else 'mp4'}"

                # Zapisz plik audio
                st.session_state["media_file_bytes"] = file_bytes
                st.session_state["media_file_extension"] = file_ext
                st.session_state["note_audio_bytes"] = file_bytes

                # Wyświetl podgląd audio
                st.audio(
                    file_bytes, format=st.session_state["media_content_type"])

                if st.button("Transkrybuj audio", key="transcribe_uploaded_audio"):
                    with st.spinner("Transkrybuję audio..."):
                        st.session_state["note_audio_text"] = transcribe_audio(
                            file_bytes)
                        st.success("Transkrypcja zakończona!")

                if st.session_state.get("note_audio_text"):
                    st.session_state["note_text"] = st.text_area(
                        "Edytuj notatkę", value=st.session_state["note_audio_text"])

            elif is_video:
                st.session_state["source_type"] = "video"
                st.session_state["media_content_type"] = f"video/{file_ext}"

                # Zapisz plik wideo
                st.session_state["media_file_bytes"] = file_bytes
                st.session_state["media_file_extension"] = file_ext

                # Wyświetl podgląd wideo
                st.video(file_bytes)

                if st.button("Ekstraktuj audio i transkrybuj", key="transcribe_video"):
                    with st.spinner("Ekstraktuję audio z wideo..."):
                        try:
                            audio_bytes = extract_audio_from_video(file_bytes)
                            st.session_state["note_audio_bytes"] = audio_bytes
                            st.success("Audio wyekstraktowane!")
                        except Exception as e:
                            st.error(
                                f"Błąd podczas ekstrakcji audio: {str(e)}")
                            st.stop()

                    with st.spinner("Transkrybuję audio..."):
                        # Dla wideo pobierz też timestampy
                        transcript_data = transcribe_audio(
                            st.session_state["note_audio_bytes"],
                            include_timestamps=True
                        )
                        st.session_state["note_audio_text"] = transcript_data["text"]
                        st.session_state["note_timestamps"] = transcript_data["segments"]
                        st.success("Transkrypcja zakończona!")

                if st.session_state.get("note_audio_text"):
                    st.session_state["note_text"] = st.text_area(
                        "Edytuj notatkę", value=st.session_state["note_audio_text"])

                    # Pokaż timestampy dla wideo
                    if st.session_state.get("note_timestamps"):
                        with st.expander("📌 Zobacz transkrypcję z timestampami"):
                            for segment in st.session_state["note_timestamps"]:
                                start_min = int(segment["start"] // 60)
                                start_sec = int(segment["start"] % 60)
                                timestamp_str = f"{start_min:02d}:{start_sec:02d}"
                                st.markdown(
                                    f"**[{timestamp_str}]** {segment['text']}")

            else:
                st.error(f"Nieobsługiwany format pliku: {file_ext}")

    elif source_option == "🖼️ Upload obrazków":
        st.session_state["source_type"] = "image"
        
        # Upload wielu obrazków
        uploaded_images = st.file_uploader(
            "Wybierz obrazki (możesz wybrać wiele)",
            type=["png", "jpg", "jpeg", "gif", "bmp", "tiff", "webp"],
            accept_multiple_files=True,
            help="Obsługiwane formaty: PNG, JPG, JPEG, GIF, BMP, TIFF, WEBP"
        )
        
        if uploaded_images:
            st.write(f"📸 Wybrano **{len(uploaded_images)}** obrazków")
            
            # Pokaż podgląd obrazków
            if len(uploaded_images) <= 6:  # Pokaż wszystkie jeśli mało
                cols = st.columns(min(len(uploaded_images), 3))
                for i, img in enumerate(uploaded_images):
                    with cols[i % 3]:
                        st.image(img, caption=img.name, use_column_width=True)
            else:  # Pokaż tylko pierwsze 6
                st.write("**Podgląd pierwszych 6 obrazków:**")
                cols = st.columns(3)
                for i, img in enumerate(uploaded_images[:6]):
                    with cols[i % 3]:
                        st.image(img, caption=img.name, use_column_width=True)
                st.info(f"... i {len(uploaded_images) - 6} więcej")
            
            # Przetwarzanie obrazków
            if st.button("🔍 Odczytać tekst z obrazków", type="primary"):
                if len(uploaded_images) == 1:
                    # Pojedynczy obrazek
                    with st.spinner("Przetwarzam obrazek..."):
                        try:
                            image_bytes = uploaded_images[0].read()
                            st.session_state["media_file_bytes"] = image_bytes
                            st.session_state["media_file_extension"] = uploaded_images[0].name.split('.')[-1].lower()
                            st.session_state["media_content_type"] = f"image/{st.session_state['media_file_extension']}"
                            
                            extracted_text = extract_text_from_image(image_bytes)
                            st.session_state["note_audio_text"] = extracted_text
                            st.session_state["note_text"] = extracted_text
                            
                            if extracted_text:
                                st.success("✅ Tekst wyekstraktowany pomyślnie!")
                            else:
                                st.warning("⚠️ Nie udało się odczytać tekstu z obrazka")
                                
                        except Exception as e:
                            st.error(f"❌ Błąd: {str(e)}")
                else:
                    # Wiele obrazków
                    with st.spinner(f"Przetwarzam {len(uploaded_images)} obrazków..."):
                        try:
                            result = process_multiple_images(uploaded_images)
                            
                            # Zapisz dane pierwszego obrazka jako główny plik
                            if result['image_data']:
                                st.session_state["media_file_bytes"] = result['image_data'][0]['bytes']
                                st.session_state["media_file_extension"] = result['image_data'][0]['extension']
                                st.session_state["media_content_type"] = result['image_data'][0]['content_type']
                            
                            # Ustaw tekst
                            st.session_state["note_audio_text"] = result['combined_text']
                            st.session_state["note_text"] = result['combined_text']
                            
                            # Zapisz dane o wszystkich obrazkach
                            st.session_state["multiple_images_data"] = result['image_data']
                            st.session_state["individual_texts"] = result['individual_texts']
                            
                            st.success(f"✅ Przetworzono {len(uploaded_images)} obrazków!")
                            
                        except Exception as e:
                            st.error(f"❌ Błąd: {str(e)}")
            
            # Wyświetl wyekstraktowany tekst
            if st.session_state.get("note_text"):
                st.markdown("### 📝 Wyekstraktowany tekst")
                
                # Dla wielu obrazków pokaż szczegóły
                if st.session_state.get("individual_texts") and len(st.session_state["individual_texts"]) > 1:
                    with st.expander("📋 Zobacz tekst z każdego obrazka osobno"):
                        for item in st.session_state["individual_texts"]:
                            st.markdown(f"**{item['filename']}:**")
                            if item['text']:
                                st.text_area("", value=item['text'], height=100, key=f"text_{item['filename']}", disabled=True)
                            else:
                                st.info("Brak tekstu do odczytania")
                            st.divider()
                
                # Edytowalny tekst
                st.session_state["note_text"] = st.text_area(
                    "Edytuj wyekstraktowany tekst",
                    value=st.session_state["note_text"],
                    height=300
                )

    elif source_option == "📱 Pobierz z Instagram":
        st.session_state["source_type"] = "instagram"

        st.info("ℹ️ Obsługiwane: Reels, Posty z wideo, IGTV")

        instagram_url = st.text_input(
            "🔗 URL do rolki/wideo Instagram",
            placeholder="https://www.instagram.com/reel/...",
            help="Wklej link do rolki, posta z wideo lub IGTV"
        )

        if instagram_url and st.button("📥 Pobierz i przetwórz", type="primary"):
            # Walidacja URL
            if not is_instagram_url(instagram_url):
                st.error(
                    "❌ To nie wygląda na prawidłowy URL Instagram. Sprawdź link!")
                st.stop()

            try:
                # Pobieranie wideo
                with st.spinner("📥 Pobieram wideo z Instagram..."):
                    video_data = download_instagram_video(instagram_url)
                    st.success("✅ Wideo pobrane!")

                # Zapisz dane wideo
                st.session_state["media_file_bytes"] = video_data["video_bytes"]
                st.session_state["media_file_extension"] = "mp4"
                st.session_state["media_content_type"] = "video/mp4"
                st.session_state["instagram_metadata"] = {
                    "title": video_data["title"],
                    "author": video_data["author"],
                    "description": video_data["description"],
                    "url": video_data["url"],
                    "duration": video_data["duration"]
                }

                # Pokaż metadane
                st.markdown("### 📊 Informacje o wideo")
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Autor:** {video_data['author']}")
                    st.write(f"**Długość:** {video_data['duration']}s")
                with col2:
                    st.write(f"**Tytuł:** {video_data['title'][:50]}...")

                if video_data.get("description"):
                    with st.expander("📝 Zobacz opis"):
                        st.write(video_data["description"])

                # Podgląd wideo
                st.video(video_data["video_bytes"])

                # Ekstrakcja audio i transkrypcja
                with st.spinner("🎵 Ekstraktuję audio z wideo..."):
                    audio_bytes = extract_audio_from_video(
                        video_data["video_bytes"])
                    st.session_state["note_audio_bytes"] = audio_bytes
                    st.success("✅ Audio wyekstraktowane!")

                with st.spinner("✍️ Transkrybuję audio (może to chwilę potrwać)..."):
                    transcript_data = transcribe_audio(
                        audio_bytes,
                        include_timestamps=True
                    )
                    st.session_state["note_audio_text"] = transcript_data["text"]
                    st.session_state["note_timestamps"] = transcript_data["segments"]
                    st.success("✅ Transkrypcja zakończona!")

            except Exception as e:
                st.error(f"❌ Błąd: {str(e)}")
                st.info(
                    "💡 Wskazówka: Upewnij się, że profil nie jest prywatny lub spróbuj innego linku")
                st.stop()

        # Wyświetl transkrypcję i pozwól na edycję
        if st.session_state.get("note_audio_text"):
            st.markdown("### 📝 Transkrypcja")

            # Dodaj metadane Instagram do tekstu notatki (opcjonalnie)
            if st.session_state.get("instagram_metadata"):
                meta = st.session_state["instagram_metadata"]
                default_text = f"🎥 Instagram: {meta['author']}\n🔗 {meta['url']}\n\n{st.session_state['note_audio_text']}"
            else:
                default_text = st.session_state["note_audio_text"]

            st.session_state["note_text"] = st.text_area(
                "Edytuj notatkę",
                value=default_text,
                height=300
            )

            # Pokaż timestampy
            if st.session_state.get("note_timestamps"):
                with st.expander("📌 Zobacz transkrypcję z timestampami"):
                    for segment in st.session_state["note_timestamps"]:
                        start_min = int(segment["start"] // 60)
                        start_sec = int(segment["start"] % 60)
                        timestamp_str = f"{start_min:02d}:{start_sec:02d}"
                        st.markdown(f"**[{timestamp_str}]** {segment['text']}")

    else:  # Napisz tekst
        st.session_state["source_type"] = "text"
        st.session_state["media_file_bytes"] = None  # Brak pliku dla tekstu
        st.session_state["note_text"] = st.text_area(
            "Wprowadź treść notatki",
            value=st.session_state.get("note_text", ""),
            height=200
        )

    # Dodawanie kategorii
    st.divider()
    st.subheader("Katalogowanie notatki")

    # Inicjalizuj note_categories jeśli nie istnieje
    if "note_categories" not in st.session_state:
        st.session_state["note_categories"] = []

    col1, col2 = st.columns([3, 1])

    with col1:
        existing_categories = get_all_categories()

        # Multiselect ze wszystkimi kategoriami (istniejące + nowe dodane lokalnie)
        all_available_categories = list(
            set(existing_categories + st.session_state["note_categories"]))

        selected_categories = st.multiselect(
            "Wybierz kategorie",
            options=all_available_categories,
            default=st.session_state["note_categories"],
            key="category_multiselect"
        )

        # Aktualizuj session_state na podstawie wyboru
        st.session_state["note_categories"] = selected_categories

    with col2:
        new_category = st.text_input(
            "Nowa kategoria", key="new_category_input")
        if st.button("➕ Dodaj", key="add_category_btn") and new_category:
            if new_category not in st.session_state["note_categories"]:
                st.session_state["note_categories"].append(new_category)
                st.rerun()

    if st.session_state["note_categories"]:
        st.write("Wybrane kategorie:", ", ".join(
            st.session_state["note_categories"]))

    # Zapisz notatkę
    st.divider()

    # Opcja zapisu pliku do Spaces
    save_media = False
    if st.session_state.get("media_file_bytes") and get_spaces_client():
        save_media = st.checkbox(
            "💾 Zapisz oryginalny plik audio/wideo w DigitalOcean Spaces",
            value=True,
            help="Plik będzie dostępny do odtworzenia w przyszłości"
        )

    if st.session_state.get("note_text") and st.button(
        "💾 Zapisz notatkę",
        disabled=not st.session_state.get("note_text"),
        type="primary"
    ):
        media_url = None
        media_type = None

        # Upload pliku do Spaces jeśli wybrano
        if save_media and st.session_state.get("media_file_bytes"):
            with st.spinner("Uploaduję plik do Spaces..."):
                media_url = upload_file_to_spaces(
                    st.session_state["media_file_bytes"],
                    st.session_state.get("media_file_extension", "mp3"),
                    st.session_state.get("media_content_type", "audio/mp3")
                )
                if media_url:
                    media_type = st.session_state.get("media_content_type")
                    st.success("✅ Plik zapisany w Spaces")
                else:
                    st.warning(
                        "⚠️ Nie udało się zapisać pliku w Spaces, ale notatka zostanie zapisana bez załącznika")

        # Zapisz notatkę do bazy
        add_note_to_db(
            note_text=st.session_state["note_text"],
            categories=st.session_state["note_categories"],
            source_type=st.session_state["source_type"],
            media_url=media_url,
            media_type=media_type,
            timestamps=st.session_state.get("note_timestamps"),  # Timestampy dla wideo/audio
            multiple_images_data=st.session_state.get("multiple_images_data")  # Dane o obrazkach
        )
        st.toast("Notatka zapisana", icon="🎉")

        # Reset state
        st.session_state["note_text"] = ""
        st.session_state["note_audio_text"] = ""
        st.session_state["note_categories"] = []
        st.session_state["media_file_bytes"] = None
        st.session_state["note_timestamps"] = None  # Wyczyść timestampy
        st.session_state["multiple_images_data"] = None  # Wyczyść dane o obrazkach
        st.session_state["individual_texts"] = None  # Wyczyść indywidualne teksty
        st.rerun()


with search_tab:
    st.subheader("Wyszukaj notatki")

    # Formularz wyszukiwania - Enter automatycznie uruchamia wyszukiwanie
    with st.form("search_form"):
        query = st.text_input("Wyszukaj notatkę",
                              placeholder="Wpisz słowa kluczowe...")

        col1, col2 = st.columns(2)

        with col1:
            categories = get_all_categories()
            selected_category = st.selectbox(
                "Filtruj po kategorii",
                options=["Wszystkie"] + categories
            )

        with col2:
            show_context = st.checkbox(
                "🤖 Pokaż kontekst AI",
                value=True,
                help="Generuje wyjaśnienie dlaczego notatka pasuje do zapytania (wymaga API OpenAI)"
            )

        search_submitted = st.form_submit_button("🔍 Szukaj", type="primary")

    if search_submitted:
        category_filter = selected_category if selected_category != "Wszystkie" else None
        notes = list_notes_from_db(query, category_filter=category_filter)

        if notes:
            st.write(f"Znaleziono: **{len(notes)}** notatek")

            # Generuj konteksty dla wszystkich wyników (z progress bar)
            if query and show_context:  # Tylko jeśli było wyszukiwanie tekstowe i opcja włączona
                with st.spinner("Analizuję wyniki za pomocą AI..."):
                    for note_item in notes:
                        context = generate_search_context(
                            query, note_item.get("text", ""))
                        note_item["search_context"] = context

            for idx, note_item in enumerate(notes):
                note_id = note_item.get("id")
                qdrant_id = note_item.get("qdrant_id")
                note_key = f"search_{idx}"

                with st.container(border=True):
                    # Nagłówek z metadanymi
                    col1, col2, col3, col4 = st.columns([2, 1, 1, 1])
                    with col1:
                        if note_item.get("timestamp"):
                            dt = datetime.fromisoformat(note_item["timestamp"])
                            st.caption(f"📅 {dt.strftime('%Y-%m-%d %H:%M')}")
                    with col2:
                        source_icons = {"audio": "🎤",
                                        "video": "🎥", "text": "📝", "image": "🖼️", "instagram": "📱"}
                        icon = source_icons.get(
                            note_item.get("source_type", "audio"), "📝")
                        st.caption(
                            f"{icon} {note_item.get('source_type', 'audio')}")
                    with col3:
                        if note_item.get("score"):
                            st.caption(f"⭐ {note_item['score']:.4f}")
                    with col4:
                        # Przyciski akcji
                        action_col1, action_col2 = st.columns(2)
                        with action_col1:
                            if st.button("✏️", key=f"edit_{note_key}", help="Edytuj notatkę"):
                                st.session_state[f"editing_{note_key}"] = True
                        with action_col2:
                            if st.button("🗑️", key=f"delete_{note_key}", help="Usuń notatkę", type="secondary"):
                                if delete_note(note_id, qdrant_id):
                                    st.success("Notatka usunięta!")
                                    st.rerun()

                    # Kontekst wyszukiwania (jeśli dostępny)
                    if note_item.get("search_context"):
                        st.info(
                            f"🔍 **Dlaczego pasuje:** {note_item['search_context']}")

                    # Media player jeśli plik jest dostępny
                    if note_item.get("media_url"):
                        media_type = note_item.get("media_type", "")
                        if media_type.startswith("audio"):
                            st.audio(note_item["media_url"])
                        elif media_type.startswith("video"):
                            st.video(note_item["media_url"])
                        elif media_type.startswith("image"):
                            st.image(note_item["media_url"])

                    # Timestampy dla wideo/audio
                    if note_item.get("timestamps") and note_item.get("source_type") in ["video", "audio"]:
                        with st.expander("📌 Transkrypcja z timestampami"):
                            for segment in note_item["timestamps"]:
                                start_min = int(segment["start"] // 60)
                                start_sec = int(segment["start"] % 60)
                                timestamp_str = f"{start_min:02d}:{start_sec:02d}"
                                st.markdown(
                                    f"**[{timestamp_str}]** {segment['text']}")

                    # Edycja notatki
                    if st.session_state.get(f"editing_{note_key}", False):
                        st.divider()

                        # Edycja treści
                        new_text = st.text_area(
                            "Edytuj treść notatki",
                            value=note_item.get("text", ""),
                            height=150,
                            key=f"text_{note_key}"
                        )

                        # Edycja kategorii
                        all_cats = get_all_categories()
                        new_categories = st.multiselect(
                            "Zmień kategorie",
                            options=all_cats,
                            default=note_item.get("categories", []),
                            key=f"cats_{note_key}"
                        )

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("💾 Zapisz zmiany", key=f"save_{note_key}", type="primary"):
                                success = True
                                # Aktualizuj tekst jeśli się zmienił
                                if new_text != note_item.get("text"):
                                    success = success and update_note_text(
                                        note_id, new_text)
                                # Aktualizuj kategorie jeśli się zmieniły
                                if set(new_categories) != set(note_item.get("categories", [])):
                                    success = success and update_note_categories(
                                        note_id, new_categories)

                                if success:
                                    st.success("Notatka zaktualizowana!")
                                    st.session_state[f"editing_{note_key}"] = False
                                    st.rerun()
                        with col_cancel:
                            if st.button("❌ Anuluj", key=f"cancel_{note_key}"):
                                st.session_state[f"editing_{note_key}"] = False
                                st.rerun()
                    else:
                        # Wyświetl wiele obrazków jeśli dostępne
                        if note_item.get("source_type") == "image" and note_item.get("multiple_images_data"):
                            with st.expander("🖼️ Zobacz wszystkie obrazki"):
                                images_data = note_item["multiple_images_data"]
                                if len(images_data) <= 6:
                                    cols = st.columns(min(len(images_data), 3))
                                    for i, img_data in enumerate(images_data):
                                        with cols[i % 3]:
                                            st.image(img_data["bytes"], caption=img_data["name"], use_column_width=True)
                                else:
                                    st.write("**Pierwsze 6 obrazków:**")
                                    cols = st.columns(3)
                                    for i, img_data in enumerate(images_data[:6]):
                                        with cols[i % 3]:
                                            st.image(img_data["bytes"], caption=img_data["name"], use_column_width=True)
                                    st.info(f"... i {len(images_data) - 6} więcej")
                        
                        # Treść notatki z podświetleniem (tylko w wyszukiwaniu)
                        if query:
                            highlighted_text = highlight_text(
                                note_item["text"], query)
                            st.markdown(highlighted_text,
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(note_item["text"])

                        # Wyświetl kategorie
                        if note_item.get("categories"):
                            tags = " ".join(
                                [f"`{cat}`" for cat in note_item["categories"]])
                            st.markdown(f"🏷️ {tags}")
        else:
            st.info("Nie znaleziono notatek")


with browse_tab:
    st.subheader("Przeglądaj wszystkie notatki")

    # Filtry
    col1, col2, col3 = st.columns(3)

    with col1:
        categories = get_all_categories()
        filter_category = st.selectbox(
            "Kategoria",
            options=["Wszystkie"] + categories,
            key="browse_category"
        )

    with col2:
        date_from = st.date_input(
            "Data od",
            value=None,
            key="browse_date_from"
        )

    with col3:
        date_to = st.date_input(
            "Data do",
            value=None,
            key="browse_date_to"
        )

    # Pobierz notatki
    category_filter = filter_category if filter_category != "Wszystkie" else None
    notes = list_notes_from_db(
        category_filter=category_filter,
        date_from=date_from,
        date_to=date_to,
        limit=100
    )

    st.write(f"Wyświetlam: **{len(notes)}** notatek")

    # Grupowanie po dacie
    if notes:
        notes_by_date = {}
        for note in notes:
            if note.get("timestamp"):
                dt = datetime.fromisoformat(note["timestamp"])
                date_key = dt.strftime("%Y-%m-%d")
                if date_key not in notes_by_date:
                    notes_by_date[date_key] = []
                notes_by_date[date_key].append(note)

        # Wyświetl pogrupowane notatki
        for date_key in sorted(notes_by_date.keys(), reverse=True):
            st.subheader(f"📅 {date_key}")
            day_notes = notes_by_date[date_key]

            for idx, note_item in enumerate(day_notes):
                note_id = note_item.get("id")
                qdrant_id = note_item.get("qdrant_id")
                note_key = f"{date_key}_{idx}"

                with st.container(border=True):
                    # Nagłówek z metadanymi
                    col1, col2, col3 = st.columns([2, 1, 1])
                    with col1:
                        if note_item.get("timestamp"):
                            dt = datetime.fromisoformat(note_item["timestamp"])
                            st.caption(f"🕐 {dt.strftime('%H:%M')}")
                    with col2:
                        source_icons = {"audio": "🎤",
                                        "video": "🎥", "text": "📝", "image": "🖼️", "instagram": "📱"}
                        icon = source_icons.get(
                            note_item.get("source_type", "audio"), "📝")
                        st.caption(
                            f"{icon} {note_item.get('source_type', 'audio')}")
                    with col3:
                        # Przyciski akcji
                        action_col1, action_col2 = st.columns(2)
                        with action_col1:
                            if st.button("✏️", key=f"edit_{note_key}", help="Edytuj notatkę"):
                                st.session_state[f"editing_{note_key}"] = True
                        with action_col2:
                            if st.button("🗑️", key=f"delete_{note_key}", help="Usuń notatkę", type="secondary"):
                                if delete_note(note_id, qdrant_id):
                                    st.success("Notatka usunięta!")
                                    st.rerun()

                    # Media player jeśli plik jest dostępny
                    if note_item.get("media_url"):
                        media_type = note_item.get("media_type", "")
                        if media_type.startswith("audio"):
                            st.audio(note_item["media_url"])
                        elif media_type.startswith("video"):
                            st.video(note_item["media_url"])
                        elif media_type.startswith("image"):
                            st.image(note_item["media_url"])

                    # Timestampy dla wideo/audio
                    if note_item.get("timestamps") and note_item.get("source_type") in ["video", "audio"]:
                        with st.expander("📌 Transkrypcja z timestampami"):
                            for segment in note_item["timestamps"]:
                                start_min = int(segment["start"] // 60)
                                start_sec = int(segment["start"] % 60)
                                timestamp_str = f"{start_min:02d}:{start_sec:02d}"
                                st.markdown(
                                    f"**[{timestamp_str}]** {segment['text']}")

                    # Edycja notatki
                    if st.session_state.get(f"editing_{note_key}", False):
                        st.divider()

                        # Edycja treści
                        new_text = st.text_area(
                            "Edytuj treść notatki",
                            value=note_item.get("text", ""),
                            height=150,
                            key=f"text_{note_key}"
                        )

                        # Edycja kategorii
                        all_cats = get_all_categories()
                        new_categories = st.multiselect(
                            "Zmień kategorie",
                            options=all_cats,
                            default=note_item.get("categories", []),
                            key=f"cats_{note_key}"
                        )

                        col_save, col_cancel = st.columns(2)
                        with col_save:
                            if st.button("💾 Zapisz zmiany", key=f"save_{note_key}", type="primary"):
                                success = True
                                # Aktualizuj tekst jeśli się zmienił
                                if new_text != note_item.get("text"):
                                    success = success and update_note_text(
                                        note_id, new_text)
                                # Aktualizuj kategorie jeśli się zmieniły
                                if set(new_categories) != set(note_item.get("categories", [])):
                                    success = success and update_note_categories(
                                        note_id, new_categories)

                                if success:
                                    st.success("Notatka zaktualizowana!")
                                    st.session_state[f"editing_{note_key}"] = False
                                    st.rerun()
                        with col_cancel:
                            if st.button("❌ Anuluj", key=f"cancel_{note_key}"):
                                st.session_state[f"editing_{note_key}"] = False
                                st.rerun()
                    else:
                        # Wyświetl wiele obrazków jeśli dostępne
                        if note_item.get("source_type") == "image" and note_item.get("multiple_images_data"):
                            with st.expander("🖼️ Zobacz wszystkie obrazki"):
                                images_data = note_item["multiple_images_data"]
                                if len(images_data) <= 6:
                                    cols = st.columns(min(len(images_data), 3))
                                    for i, img_data in enumerate(images_data):
                                        with cols[i % 3]:
                                            st.image(img_data["bytes"], caption=img_data["name"], use_column_width=True)
                                else:
                                    st.write("**Pierwsze 6 obrazków:**")
                                    cols = st.columns(3)
                                    for i, img_data in enumerate(images_data[:6]):
                                        with cols[i % 3]:
                                            st.image(img_data["bytes"], caption=img_data["name"], use_column_width=True)
                                    st.info(f"... i {len(images_data) - 6} więcej")
                        
                        # Treść notatki
                        st.markdown(note_item["text"])

                        # Wyświetl kategorie
                        if note_item.get("categories"):
                            tags = " ".join(
                                [f"`{cat}`" for cat in note_item["categories"]])
                            st.markdown(f"🏷️ {tags}")
    else:
        st.info("Brak notatek do wyświetlenia")


with manage_categories_tab:
    st.subheader("Zarządzaj kategoriami")

    # Sekcja dodawania nowej kategorii
    st.markdown("### ➕ Dodaj nową kategorię")
    col1, col2 = st.columns([3, 1])

    with col1:
        new_category_name = st.text_input(
            "Nazwa nowej kategorii",
            key="manage_new_category",
            placeholder="Wprowadź nazwę kategorii..."
        )

    with col2:
        st.write("")  # spacer
        st.write("")  # spacer
        if st.button("➕ Dodaj kategorię", key="add_new_category_btn", type="primary"):
            if new_category_name and new_category_name.strip():
                conn = get_postgres_connection()
                if conn:
                    try:
                        with conn.cursor() as cur:
                            cur.execute("""
                                INSERT INTO categories (name)
                                VALUES (%s)
                                ON CONFLICT (name) DO NOTHING
                                RETURNING id
                            """, (new_category_name.strip(),))
                            result = cur.fetchone()

                            if result:
                                conn.commit()
                                st.success(
                                    f"✅ Kategoria '{new_category_name}' dodana!")
                                st.rerun()
                            else:
                                st.warning(
                                    f"⚠️ Kategoria '{new_category_name}' już istnieje")
                    except Exception as e:
                        conn.rollback()
                        st.error(f"Błąd dodawania kategorii: {str(e)}")
                    finally:
                        conn.close()
                else:
                    st.error("Wymaga połączenia z PostgreSQL")
            else:
                st.warning("Wprowadź nazwę kategorii")

    st.divider()

    # Pobierz statystyki kategorii
    categories_stats = get_category_stats()

    if not categories_stats:
        st.info("Brak kategorii. Dodaj pierwszą kategorię powyżej.")
    else:
        st.markdown(f"### 📋 Lista kategorii ({len(categories_stats)})")

        # Tabela z kategoriami
        for cat in categories_stats:
            cat_id = cat["id"]
            cat_name = cat["name"]
            note_count = cat["note_count"]

            with st.container(border=True):
                col1, col2, col3 = st.columns([3, 1, 1])

                with col1:
                    # Edycja nazwy
                    if st.session_state.get(f"editing_cat_{cat_id}", False):
                        new_name = st.text_input(
                            "Nowa nazwa kategorii",
                            value=cat_name,
                            key=f"catname_{cat_id}"
                        )
                    else:
                        st.markdown(f"### {cat_name}")
                        st.caption(f"Użyta w **{note_count}** notatkach")

                with col2:
                    # Przycisk edycji
                    if st.session_state.get(f"editing_cat_{cat_id}", False):
                        if st.button("💾 Zapisz", key=f"save_cat_{cat_id}", type="primary"):
                            if new_name and new_name != cat_name:
                                if rename_category(cat_id, new_name):
                                    st.success(
                                        f"Zmieniono nazwę na '{new_name}'")
                                    st.session_state[f"editing_cat_{cat_id}"] = False
                                    st.rerun()
                            else:
                                st.session_state[f"editing_cat_{cat_id}"] = False
                                st.rerun()
                    else:
                        if st.button("✏️ Zmień", key=f"edit_cat_{cat_id}"):
                            st.session_state[f"editing_cat_{cat_id}"] = True
                            st.rerun()

                with col3:
                    # Przycisk usuwania
                    if st.session_state.get(f"editing_cat_{cat_id}", False):
                        if st.button("❌ Anuluj", key=f"cancel_cat_{cat_id}"):
                            st.session_state[f"editing_cat_{cat_id}"] = False
                            st.rerun()
                    else:
                        if st.button("🗑️ Usuń", key=f"delete_cat_{cat_id}", type="secondary"):
                            if note_count > 0:
                                st.warning(
                                    f"⚠️ Ta kategoria jest użyta w {note_count} notatkach. Usunięcie kategorii usunie ją ze wszystkich notatek.")
                                col_confirm, col_cancel = st.columns(2)
                                with col_confirm:
                                    if st.button("✅ Potwierdź usunięcie", key=f"confirm_delete_{cat_id}", type="primary"):
                                        if delete_category(cat_id):
                                            st.success("Kategoria usunięta!")
                                            st.rerun()
                                with col_cancel:
                                    if st.button("❌ Anuluj", key=f"cancel_delete_{cat_id}"):
                                        st.rerun()
                            else:
                                if delete_category(cat_id):
                                    st.success("Kategoria usunięta!")
                                    st.rerun()
