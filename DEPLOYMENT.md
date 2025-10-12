# 🚀 Instrukcja Wdrożenia na Streamlit Cloud

## Krok 1: Przygotowanie repozytorium

1. Upewnij się, że wszystkie pliki są w repozytorium:

   - `app.py`
   - `requirements.txt`
   - `packages.txt` (zawiera: ffmpeg)
   - `.gitignore`
   - `README.md`

2. Sprawdź plik `.gitignore`:

```
.env
__pycache__/
```

3. Stwórz plik `.streamlit/secrets.toml` lokalnie (NIE commituj do repo!):

```toml
OPENAI_API_KEY = "sk-your-openai-api-key"
QDRANT_URL = "https://your-qdrant-instance.cloud.qdrant.io"
QDRANT_API_KEY = "your-qdrant-api-key"

# Opcjonalne - dla przechowywania mediów
DO_SPACES_KEY = "your-digitalocean-spaces-key"
DO_SPACES_SECRET = "your-digitalocean-spaces-secret"
DO_SPACES_REGION = "fra1"
DO_SPACES_BUCKET = "your-bucket-name"
```

## Krok 2: Push do GitHuba

```bash
git add .
git commit -m "Add audio/video notes app with DigitalOcean Spaces integration"
git push origin main
```

## Krok 3: Konfiguracja Streamlit Cloud

1. Przejdź do [share.streamlit.io](https://share.streamlit.io)
2. Kliknij **New app**
3. Wybierz swoje repozytorium
4. Ustaw:
   - **Main file path**: `modul7/app_note_final/app.py`
   - **Python version**: 3.11

## Krok 4: Dodanie Secrets

1. W ustawieniach aplikacji znajdź **Secrets**
2. Wklej zawartość z lokalnego `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-actual-api-key"
QDRANT_URL = "https://your-actual-qdrant-url"
QDRANT_API_KEY = "your-actual-qdrant-key"

# Opcjonalne
DO_SPACES_KEY = "your-actual-spaces-key"
DO_SPACES_SECRET = "your-actual-spaces-secret"
DO_SPACES_REGION = "fra1"
DO_SPACES_BUCKET = "your-actual-bucket-name"
```

3. Zapisz

## Krok 5: Deploy

1. Kliknij **Deploy**
2. Poczekaj na instalację zależności (~2-3 minuty)
3. Aplikacja powinna się uruchomić!

## 🔧 Troubleshooting

### Błąd: ModuleNotFoundError

**Problem**: Brak któregoś z modułów

**Rozwiązanie**:

- Sprawdź czy `requirements.txt` ma wszystkie pakiety
- Sprawdź czy nazwa pliku to `requirements.txt` (nie `requrements.txt`!)

### Błąd: ffmpeg not found

**Problem**: Brak ffmpeg (potrzebny do wideo)

**Rozwiązanie**:

- Sprawdź czy masz plik `packages.txt` z zawartością: `ffmpeg`

### Błąd: No module named 'audiorecorder'

**Problem**: Źle zainstalowany streamlit-audiorecorder

**Rozwiązanie**:

- Upewnij się że w `requirements.txt` jest: `streamlit-audiorecorder` (bez spacji)

### Błąd połączenia z DigitalOcean Spaces

**Problem**: Nieprawidłowe dane dostępowe

**Rozwiązanie**:

- Sprawdź czy secrets są poprawnie ustawione
- Sprawdź czy bucket istnieje
- Sprawdź czy region jest poprawny
- Aplikacja działa bez Spaces - po prostu nie będzie opcji zapisu plików

## 📝 Uwagi

- Secrets są zaszyfrowane i bezpieczne
- Secrets NIE są widoczne w kodzie ani logach
- Aplikacja automatycznie restartuje się po zmianie secrets
- DigitalOcean Spaces jest OPCJONALNY - aplikacja działa bez niego

## 🎯 Po wdrożeniu

1. Przetestuj każdą funkcjonalność:

   - ✅ Nagrywanie audio
   - ✅ Upload wideo
   - ✅ Tekstowe notatki
   - ✅ Transkrypcja
   - ✅ Dodawanie kategorii
   - ✅ Wyszukiwanie
   - ✅ Przeglądanie po datach
   - ✅ Zapisywanie w Spaces (jeśli skonfigurowane)

2. Sprawdź logi w Streamlit Cloud w razie problemów

3. Udostępnij link aplikacji!
