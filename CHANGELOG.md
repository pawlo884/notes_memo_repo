# Changelog - Audio & Video Notes App

## 🔧 Hotfix 3.0.1 - Python 3.11 Compatibility

### 🐛 Poprawki

- ✅ **WYMAGANY Python 3.11** - dodano `runtime.txt` z `3.11.9`
- ✅ **Kompatybilność z pydub** - Python 3.13+ ma problemy z modułem `pyaudioop` 
- ✅ **Streamlit Cloud** - aplikacja WYMAGA Python 3.11 (nie 3.13+)
- ✅ **Naprawiono błąd instalacji** - usunięto nieprawidłowy `python-version==3.11` z requirements.txt

### 📝 Techniczne

- Dodano `runtime.txt` z `3.11.9` (Streamlit Cloud używa tego do określenia wersji)
- Rozwiązano błąd: `import pyaudioop as audioop` w Python 3.13+
- **WAŻNE**: Jeśli Streamlit Cloud nadal używa 3.13, usuń aplikację i wdróż ponownie z Python 3.11

---

## 🚀 Wersja 3.0 - PostgreSQL & Rozszerzone formaty ⭐ NOWOŚĆ

### ✨ Nowe funkcjonalności

#### 1. **Integracja z PostgreSQL**

- ✅ Hybrydowa architektura: PostgreSQL (dane) + Qdrant (semantyka)
- ✅ Relacyjna baza danych dla metadanych notatek
- ✅ Tabela kategorii z relacjami many-to-many
- ✅ Pełne możliwości SQL (filtrowanie, agregacje, indeksy)
- ✅ Transakcje ACID dla integralności danych
- ✅ Timestamps (created_at, updated_at)
- ✅ Automatyczna inicjalizacja tabel
- ✅ Backward compatibility - działa bez PostgreSQL

#### 2. **Rozszerzone formaty audio**

- ✅ Upload gotowych plików audio: MP3, WAV, M4A, OGG, FLAC
- ✅ Bezpośrednia transkrypcja bez konwersji
- ✅ Inteligentne rozpoznawanie typu pliku (audio vs video)
- ✅ Wspólny uploader dla audio/wideo
- ✅ Podgląd audio przed transkrypcją

#### 3. **Ulepszona obsługa wideo**

- ✅ Dodano format WEBM
- ✅ Lepsze nazewnictwo przycisków (rozróżnienie audio/wideo)
- ✅ Optymalizacja content-type dla różnych formatów

### 🔧 Zmiany techniczne

#### Nowe zależności

- `psycopg2-binary` - PostgreSQL adapter dla Python

#### Rozszerzona konfiguracja

```env
# PostgreSQL (opcjonalne)
POSTGRES_HOST=your-db-host.db.ondigitalocean.com
POSTGRES_PORT=25060
POSTGRES_DB=defaultdb
POSTGRES_USER=doadmin
POSTGRES_PASSWORD=your-password
POSTGRES_SSLMODE=require
```

#### Struktura bazy PostgreSQL

**Tabele:**

- `notes` - główna tabela z notatkami
- `categories` - katalog kategorii
- `note_categories` - relacje many-to-many

**Widoki:**

- `notes_with_categories` - notatki z agregowanymi kategoriami

**Funkcje:**

- Automatyczna aktualizacja `updated_at`
- Inicjalizacja z `init_db.sql`

#### Nowe funkcje

- `get_postgres_connection()` - połączenie z PostgreSQL (cached)
- `init_postgres_tables()` - automatyczna inicjalizacja schematu
- Rozszerzone `add_note_to_db()` - zapis do PostgreSQL + Qdrant
- Rozszerzone `list_notes_from_db()` - hybrid querying
- Rozszerzone `get_all_categories()` - z PostgreSQL lub Qdrant

#### Model danych (PostgreSQL + Qdrant)

**PostgreSQL notes:**

```sql
id SERIAL PRIMARY KEY,
text TEXT NOT NULL,
timestamp TIMESTAMP,
source_type VARCHAR(20),
media_url TEXT,
media_type VARCHAR(50),
qdrant_id INTEGER,
created_at TIMESTAMP,
updated_at TIMESTAMP
```

**Qdrant payload:**

```python
{
    "text": str,
    "timestamp": str,
    "source_type": str,
    "categories": list,
    "media_url": str,
    "media_type": str,
    "postgres_id": int  # ⭐ NOWE
}
```

### 📚 Dokumentacja

#### Nowe pliki

- `init_db.sql` - Schemat bazy PostgreSQL
- `POSTGRES_SETUP.md` - Pełna instrukcja konfiguracji PostgreSQL
- Zaktualizowany `README.md` z informacjami o PostgreSQL
- Zaktualizowany `env_example.txt` z konfiguracją PostgreSQL

### 🎨 Zmiany w UI

- 📁 Zmiana nazwy zakładki: "🎥 Upload wideo" → "📁 Upload audio/wideo"
- Inteligentny uploader rozpoznający typ pliku
- Różne przyciski dla audio ("Transkrybuj audio") i wideo ("Ekstraktuj audio i transkrybuj")
- Lepsze komunikaty o obsługiwanych formatach

### 🐛 Naprawione błędy

- ✅ Poprawiono import `StreamlitSecretNotFoundError`
- ✅ Dodano fallback gdy brak `secrets.toml`
- ✅ Poprawiono obsługę braku PostgreSQL
- ✅ Obsługa nieznanych formatów plików

### ⚡ Wydajność

- ✅ PostgreSQL dla szybkich filtrów (kategorie, daty)
- ✅ Qdrant tylko dla semantycznego wyszukiwania
- ✅ Zmniejszone obciążenie Qdrant
- ✅ Indeksy w PostgreSQL dla timestamp, source_type

### 📊 Statystyki

- **Linie kodu**: ~955 (+275 vs v2.0)
- **Funkcje**: 15 (+3 vs v2.0)
- **Obsługiwane formaty audio**: MP3, WAV, M4A, OGG, FLAC
- **Obsługiwane formaty wideo**: MP4, MOV, AVI, MKV, WEBM
- **Tabele PostgreSQL**: 3 + 1 widok

---

## 🎉 Wersja 2.0 - Integracja z DigitalOcean Spaces

### ✨ Nowe funkcjonalności

#### 1. **Przechowywanie mediów w chmurze**

- ✅ Integracja z DigitalOcean Spaces (S3-compatible storage)
- ✅ Opcjonalny zapis oryginalnych plików audio/wideo
- ✅ Publiczne URLe do plików
- ✅ Automatyczne generowanie unikalnych nazw (UUID)
- ✅ Organizacja plików w strukturze `notes/{uuid}.{extension}`

#### 2. **Obsługa wideo**

- ✅ Upload plików wideo (MP4, MOV, AVI, MKV)
- ✅ Automatyczna ekstrakcja audio z wideo (MoviePy)
- ✅ Transkrypcja audio z wideo
- ✅ Zapisywanie oryginalnego pliku wideo w Spaces
- ✅ Podgląd wideo przed transkrypcją
- ✅ Odtwarzanie wideo w interfejsie po zapisie

#### 3. **Rozszerzone katalogowanie**

- ✅ Timestamp dla każdej notatki
- ✅ Typ źródła (audio/video/text)
- ✅ Wielokrotne kategorie/tagi
- ✅ URL do media w Spaces
- ✅ Typ MIME pliku

#### 4. **Ulepszony interfejs**

- ✅ Trzy zakładki: Dodaj / Szukaj / Przeglądaj
- ✅ Wybór źródła: Audio / Wideo / Tekst
- ✅ Checkbox do opcjonalnego zapisu w Spaces
- ✅ Ikony dla typów źródeł (🎤🎥📝)
- ✅ Player audio/wideo w wynikach wyszukiwania
- ✅ Player audio/wideo w widoku przeglądania
- ✅ Wizualne odznaczanie kategorii

#### 5. **Zaawansowane wyszukiwanie i filtrowanie**

- ✅ Semantyczne wyszukiwanie (OpenAI Embeddings)
- ✅ Filtrowanie po kategoriach
- ✅ Filtrowanie po zakresie dat
- ✅ Sortowanie chronologiczne
- ✅ Grupowanie notatek po dniach
- ✅ Scoring podobieństwa wyników

### 🔧 Zmiany techniczne

#### Nowe zależności

- `boto3` - AWS SDK dla Python (DigitalOcean Spaces)
- `moviepy` - przetwarzanie wideo
- `imageio-ffmpeg` - codec dla MoviePy

#### Rozszerzona konfiguracja

```env
DO_SPACES_KEY          # Klucz API Spaces
DO_SPACES_SECRET       # Secret API Spaces
DO_SPACES_REGION       # Region (np. fra1)
DO_SPACES_BUCKET       # Nazwa bucketa
```

#### Funkcje pomocnicze

- `get_spaces_client()` - klient boto3 z cache
- `upload_file_to_spaces()` - upload z ACL public-read
- `get_file_from_spaces_url()` - pobieranie pliku
- `extract_audio_from_video()` - ekstrakcja audio z wideo

#### Model danych (Qdrant payload)

```python
{
    "text": str,           # Treść notatki
    "timestamp": str,      # ISO format
    "source_type": str,    # audio/video/text
    "categories": list,    # Lista tagów
    "media_url": str,      # URL w Spaces (opcjonalnie)
    "media_type": str      # MIME type (opcjonalnie)
}
```

### 📚 Dokumentacja

#### Nowe pliki

- `README.md` - Pełna dokumentacja projektu
- `QUICKSTART.md` - Szybki start i testowanie
- `DEPLOYMENT.md` - Instrukcja wdrożenia na Streamlit Cloud
- `CHANGELOG.md` - Ten plik
- `env_example.txt` - Przykładowa konfiguracja

### 🐛 Naprawione błędy

- ✅ Poprawiono nazwę pliku: `requrements.txt` → `requirements.txt`
- ✅ Dodano obsługę błędów przy ekstrakcji audio
- ✅ Dodano sprawdzanie dostępności Spaces (opcjonalne)
- ✅ Poprawiono filtrowanie dat (cały dzień)
- ✅ Dodano reset state po zapisie notatki

### 🔒 Bezpieczeństwo

- ✅ Secrets w .env (nie commitowane)
- ✅ .gitignore dla wrażliwych danych
- ✅ Streamlit secrets dla produkcji
- ✅ Publiczne pliki tylko dla potrzeb odtwarzania

### ⚡ Wydajność

- ✅ Cache dla klientów (Qdrant, Spaces, OpenAI)
- ✅ Streaming dla dużych plików wideo
- ✅ Tymczasowe pliki automatycznie usuwane
- ✅ Lazy loading mediów

---

## Wersja 1.0 - Podstawowa funkcjonalność

### Funkcje

- ✅ Nagrywanie notatek audio
- ✅ Transkrypcja przez Whisper
- ✅ Wyszukiwanie w Qdrant
- ✅ Podstawowe kategoryzowanie

---

## 🚀 Planowane funkcje (v4.0)

### Potencjalne ulepszenia

- [ ] Edycja i usuwanie notatek ⭐ Priorytet
- [ ] Export notatek (PDF, DOCX, TXT)
- [ ] Statystyki i dashboard (wykorzystanie PostgreSQL)
- [ ] Współdzielenie notatek
- [ ] Generowanie podsumowań przez GPT-4
- [ ] Tłumaczenie notatek
- [ ] Wykrywanie języka
- [ ] Batch upload wielu plików
- [ ] Folder structure dla kategorii
- [ ] Prywatne URLe (signed URLs)
- [ ] Kompresja wideo przed zapisem
- [ ] Różne jakości audio (bitrate)
- [ ] Backup PostgreSQL do innych cloud providers
- [ ] Full-text search w PostgreSQL (dodatkowo do semantycznego)
- [ ] Mobile app (React Native)
- [ ] Voice commands
- [ ] Real-time collaboration
- [ ] Synchronizacja między urządzeniami

---

## 📊 Statystyki projektu (v3.0)

- **Linie kodu**: ~955
- **Funkcje**: 15
- **Zakładki**: 3
- **Obsługiwane formaty audio**: MP3, WAV, M4A, OGG, FLAC
- **Obsługiwane formaty wideo**: MP4, MOV, AVI, MKV, WEBM
- **Integracje**: OpenAI, Qdrant, PostgreSQL, DigitalOcean Spaces
- **Zależności**: 10 głównych pakietów
- **Bazy danych**: 2 (PostgreSQL + Qdrant)
- **Tabele SQL**: 3 + 1 widok

---

**Ostatnia aktualizacja**: 2025-10-10
