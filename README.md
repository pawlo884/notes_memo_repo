# Audio & Video Notes - Aplikacja do notatek głosowych i wideo

Rozbudowana aplikacja do tworzenia, katalogowania i wyszukiwania notatek z różnych źródeł.

## 🎯 Funkcjonalności

### 1. **Wieloźródłowe notatki**

- 🎤 **Nagrywanie audio** - bezpośrednie nagrywanie notatek głosowych
- 📁 **Upload audio/wideo** - importuj gotowe pliki:
  - **Audio**: MP3, WAV, M4A, OGG, FLAC (bezpośrednia transkrypcja)
  - **Wideo**: MP4, MOV, AVI, MKV, WEBM (automatyczna ekstrakcja audio)
- 📱 **Instagram Reels** ⭐ NOWOŚĆ - pobieranie rolek bezpośrednio z Instagram:
  - Automatyczne pobieranie wideo z linku
  - Wyciąganie metadanych (autor, tytuł, opis)
  - Transkrypcja z timestampami
  - Obsługa: Reels, posty z wideo, IGTV
- 📝 **Notatki tekstowe** - ręczne wprowadzanie tekstu

### 2. **Automatyczna transkrypcja**

- Wykorzystanie OpenAI Whisper do transkrypcji audio/wideo
- Możliwość edycji transkrypcji przed zapisem
- Wsparcie dla różnych formatów wideo

### 3. **Przechowywanie mediów w chmurze** ⭐ NOWOŚĆ

- 💾 **DigitalOcean Spaces** - przechowywanie oryginalnych plików audio/wideo
- Opcjonalny zapis plików dla późniejszego odtwarzania
- Publiczne URLe do plików
- Automatyczne odtwarzanie w interfejsie
- Bezpieczne przechowywanie w S3-kompatybilnym storage

### 4. **Katalogowanie i organizacja**

- 🏷️ Dodawanie kategorii/tagów do notatek
- Możliwość wyboru z istniejących kategorii
- Tworzenie nowych kategorii na bieżąco
- Wielokrotne kategorie dla jednej notatki

### 5. **Zaawansowane wyszukiwanie**

- 🔍 **Semantyczne wyszukiwanie** - wyszukiwanie po znaczeniu, nie tylko słowach kluczowych
- Filtrowanie po kategoriach
- Scoring podobieństwa wyników

### 6. **Przeglądanie notatek**

- 📅 **Spis treści po datach** - notatki pogrupowane chronologicznie
- Filtrowanie po:
  - Kategoriach
  - Zakresie dat (od-do)
- Sortowanie od najnowszych
- Informacje o źródle notatki (audio/video/tekst)
- Timestamp każdej notatki

## 🚀 Instalacja

### Wymagania

```bash
pip install -r requirements.txt
```

### Konfiguracja

Utwórz plik `.env` z następującymi zmiennymi (skopiuj z `env_example.txt`):

```env
# Wymagane
OPENAI_API_KEY=twój_klucz_openai
QDRANT_URL=url_do_qdrant
QDRANT_API_KEY=twój_klucz_qdrant

# PostgreSQL (opcjonalne - dla rozszerzonych funkcji) ⭐ NOWOŚĆ
POSTGRES_HOST=your-db-host.db.ondigitalocean.com
POSTGRES_PORT=25060
POSTGRES_DB=defaultdb
POSTGRES_USER=doadmin
POSTGRES_PASSWORD=twoje-haslo
POSTGRES_SSLMODE=require

# DigitalOcean Spaces (opcjonalne - dla przechowywania mediów)
DO_SPACES_KEY=twój_klucz_spaces
DO_SPACES_SECRET=twój_secret_spaces
DO_SPACES_REGION=fra1
DO_SPACES_BUCKET=nazwa_twojego_bucketa
```

Lub skonfiguruj je jako Streamlit Secrets w chmurze.

> **Uwaga**:
>
> - PostgreSQL jest **opcjonalny**. Bez niego aplikacja używa tylko Qdrant (jak wcześniej)
> - DigitalOcean Spaces też jest opcjonalny - bez niego brak przechowywania oryginalnych plików
> - Szczegóły konfiguracji PostgreSQL: zobacz `POSTGRES_SETUP.md`

## 📦 Zależności

- **openai** - transkrypcja audio przez Whisper i embeddingi
- **qdrant_client** - wektorowa baza danych do semantycznego wyszukiwania
- **psycopg2-binary** - PostgreSQL adapter dla Python
- **streamlit** - interfejs webowy
- **streamlit-audiorecorder** - nagrywanie audio
- **moviepy** - ekstrakcja audio z wideo
- **imageio-ffmpeg** - codec dla moviepy
- **pydub** - przetwarzanie audio
- **python-dotenv** - zarządzanie zmiennymi środowiskowymi
- **boto3** - integracja z DigitalOcean Spaces (S3-compatible storage)
- **yt-dlp** ⭐ NOWOŚĆ - pobieranie wideo z Instagram i innych platform
- **numpy** - operacje na embeddingach

## 💻 Uruchomienie

### Lokalnie

```bash
streamlit run app.py
```

### Streamlit Cloud

1. Push kod do GitHub
2. Połącz repozytorium ze Streamlit Cloud
3. Skonfiguruj secrets w ustawieniach aplikacji
4. Deploy!

## 📊 Struktura danych

### Hybrydowe podejście ⭐ NOWOŚĆ

Aplikacja używa **dwóch baz danych**:

1. **PostgreSQL** (główna baza relacyjna):

   - Tabela `notes` - metadane notatek
   - Tabela `categories` - katalog kategorii
   - Tabela `note_categories` - relacje many-to-many
   - Pełne możliwości SQL, transakcje, indeksy

2. **Qdrant** (baza wektorowa):
   - Embeddingi do semantycznego wyszukiwania
   - Szybkie wyszukiwanie podobieństw
   - Scoring wyników

**Każda notatka zawiera:**

- **text** - treść notatki (transkrypcja lub tekst)
- **timestamp** - data i czas utworzenia (ISO format)
- **source_type** - źródło ("audio", "video", "text")
- **categories** - lista kategorii/tagów
- **media_url** - URL do oryginalnego pliku w Spaces (opcjonalnie)
- **media_type** - typ MIME pliku (np. "audio/mp3", "video/mp4")
- **vector** - embedding dla semantycznego wyszukiwania (w Qdrant)
- **postgres_id** / **qdrant_id** - synchronizacja między bazami

**Bez PostgreSQL:** aplikacja działa jak wcześniej (tylko Qdrant).

## 🎨 Zakładki

### 1. Dodaj notatkę

- Wybór źródła:
  - 🎤 Nagrywanie audio na żywo
  - 📁 Upload gotowych plików audio (MP3, WAV, M4A, OGG, FLAC) lub wideo (MP4, MOV, AVI, MKV, WEBM)
  - 📱 Pobierz z Instagram (Reels, posty z wideo, IGTV) ⭐ NOWOŚĆ
  - 📝 Ręczne wpisanie tekstu
- Automatyczne pobieranie i przetwarzanie wideo z Instagram
- Transkrypcja audio/wideo z timestampami
- Edycja treści
- Dodawanie kategorii
- Zapis do bazy (PostgreSQL + Qdrant) z metadanymi

### 2. Szukaj notatki

- Wyszukiwanie semantyczne
- Filtrowanie po kategorii
- Wyświetlanie z metadanymi i scoringiem
- Odtwarzanie zapisanych mediów

### 3. Przeglądaj notatki

- Lista wszystkich notatek
- Filtrowanie po kategorii i datach
- Grupowanie po dniach
- Chronologiczne sortowanie
- Odtwarzanie zapisanych mediów

## 🔧 Technologie

- **Frontend**: Streamlit
- **Backend**: Python
- **AI**: OpenAI (Whisper, Embeddings)
- **Databases**:
  - PostgreSQL (główna baza relacyjna) ⭐ NOWOŚĆ
  - Qdrant (baza wektorowa do semantycznego wyszukiwania)
- **Storage**: DigitalOcean Spaces (S3-compatible)
- **Video Processing**: MoviePy
- **Cloud Integration**: boto3, psycopg2

## 🌐 Konfiguracja DigitalOcean Spaces

### Tworzenie Spaces

1. Zaloguj się do DigitalOcean
2. Przejdź do **Spaces Object Storage**
3. Kliknij **Create Space**
4. Wybierz region (np. Frankfurt - fra1)
5. Ustaw nazwę bucketa
6. Ustaw uprawnienia na **Public** (lub skonfiguruj CORS)

### Generowanie kluczy API

1. W panelu DigitalOcean przejdź do **API**
2. Kliknij **Spaces Keys**
3. **Generate New Key**
4. Skopiuj **Key** i **Secret**
5. Dodaj do pliku `.env`

### CORS (opcjonalnie)

Jeśli potrzebujesz odtwarzać media w innych domenach, skonfiguruj CORS w ustawieniach Space.

## 💡 Wskazówki

- Pliki są zapisywane z unikalnymi UUID, więc nazwy nie kolidują
- Pliki są publiczne (ACL: public-read) dla łatwego odtwarzania
- Struktura: `notes/{uuid}.{extension}`
- Możesz później zmigrować na prywatne URLe z signed URLs

## 📱 Jak używać funkcji Instagram

1. **Skopiuj link do rolki**:

   - Otwórz rolkę/wideo w aplikacji Instagram
   - Kliknij ikonę udostępniania (⋯)
   - Wybierz "Kopiuj link"

2. **Wklej w aplikacji**:

   - Wybierz zakładkę "Dodaj notatkę"
   - Kliknij "📱 Pobierz z Instagram"
   - Wklej link (np. `https://www.instagram.com/reel/...`)
   - Kliknij "📥 Pobierz i przetwórz"

3. **Automatyczne przetwarzanie**:

   - Aplikacja pobierze wideo
   - Wyświetli metadane (autor, tytuł, opis)
   - Automatycznie wyekstraktuje audio
   - Wygeneruje transkrypcję z timestampami
   - Pozwoli edytować przed zapisem

4. **Zapisane metadane**:
   - URL źródłowy (do oryginalnej rolki)
   - Autor
   - Opis (jeśli dostępny)
   - Transkrypcja z timestampami
   - Oryginalne wideo (opcjonalnie w Spaces)

**Uwaga**: Profil musi być publiczny, aby funkcja działała poprawnie.

## 📝 Licencja

Ten projekt jest częścią kursu Data Scientist.
