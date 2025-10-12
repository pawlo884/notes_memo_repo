# 🚀 Quick Start

## Testowanie lokalne (bez DigitalOcean Spaces)

### 1. Instalacja

```bash
cd modul7/app_note_final
pip install -r requirements.txt
```

### 2. Konfiguracja minimalna

Stwórz plik `.env`:

```env
OPENAI_API_KEY=sk-your-openai-api-key
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key
```

### 3. Uruchomienie

```bash
streamlit run app.py
```

Aplikacja będzie działać bez przechowywania plików - tylko transkrypcje będą zapisywane.

---

## Pełna wersja (z DigitalOcean Spaces)

### 1. Stwórz konto w DigitalOcean

Przejdź na: https://www.digitalocean.com/

### 2. Utwórz Space

1. W panelu DigitalOcean: **Create** → **Spaces Object Storage**
2. Wybierz region (np. Frankfurt - fra1)
3. Ustaw nazwę (np. `my-notes`)
4. Wybierz **Public** CDN
5. Kliknij **Create Space**

### 3. Wygeneruj klucze API

1. W panelu: **API** → **Spaces Keys** (pod Tokens/Keys)
2. Kliknij **Generate New Key**
3. Ustaw nazwę (np. "Notes App")
4. Skopiuj **Key** i **Secret** (zapisz w bezpiecznym miejscu!)

### 4. Zaktualizuj .env

```env
OPENAI_API_KEY=sk-your-openai-api-key
QDRANT_URL=https://your-qdrant-instance.cloud.qdrant.io
QDRANT_API_KEY=your-qdrant-api-key

# Nowe - DigitalOcean Spaces
DO_SPACES_KEY=YOUR_SPACES_KEY
DO_SPACES_SECRET=YOUR_SPACES_SECRET
DO_SPACES_REGION=fra1
DO_SPACES_BUCKET=my-notes
```

### 5. Uruchom aplikację

```bash
streamlit run app.py
```

Teraz możesz zapisywać oryginalne pliki audio/wideo!

---

## 📹 Pierwszy test

### Test 1: Notatka głosowa

1. Otwórz zakładkę **Dodaj notatkę**
2. Wybierz **🎤 Nagraj audio**
3. Kliknij "Nagraj notatkę"
4. Powiedz coś (np. "To jest moja pierwsza notatka testowa")
5. Zatrzymaj nagrywanie
6. Kliknij **Transkrypcja**
7. Jeśli masz Spaces: zaznacz checkbox "💾 Zapisz oryginalny plik..."
8. Kliknij **💾 Zapisz notatkę**
9. Dodaj kategorię (np. "test")
10. Kliknij **💾 Zapisz notatkę**

### Test 2: Wideo

1. Przygotuj krótkie wideo (lub pobierz testowe)
2. Wybierz **🎥 Upload wideo**
3. Wybierz plik
4. Kliknij **Ekstraktuj audio i transkrybuj**
5. Edytuj transkrypcję jeśli potrzeba
6. Dodaj kategorię
7. Zapisz

### Test 3: Wyszukiwanie

1. Przejdź do **Szukaj notatki**
2. Wpisz słowo kluczowe z notatki
3. Kliknij **🔍 Szukaj**
4. Zobacz wyniki z playerem (jeśli zapisano w Spaces)

### Test 4: Przeglądanie

1. Przejdź do **Przeglądaj notatki**
2. Zobacz notatki pogrupowane po datach
3. Odtwórz zapisane media
4. Filtruj po kategoriach i datach

---

## 💡 Tips

- **Bez Spaces**: Aplikacja działa normalnie, po prostu nie zapisuje oryginalnych plików
- **Z Spaces**: Każdy plik audio/wideo jest dostępny do odtworzenia w przyszłości
- **Koszty Spaces**: ~$5/miesiąc za 250GB + transfer
- **Transkrypcja**: Whisper kosztuje ~$0.006 za minutę audio

---

## 🔍 Troubleshooting

### "No module named X"

```bash
pip install -r requirements.txt
```

### "FFMPEG not found" (dla wideo)

```bash
# Windows (Chocolatey)
choco install ffmpeg

# macOS
brew install ffmpeg

# Linux
sudo apt install ffmpeg
```

### "Connection refused" (Qdrant)

- Sprawdź czy URL i API key są poprawne
- Sprawdź czy kolekcja jest dostępna

### "Access Denied" (Spaces)

- Sprawdź klucze API
- Sprawdź nazwę bucketa
- Sprawdź region

---

## 🎉 Gotowe!

Twoja aplikacja jest gotowa do użycia. Możesz teraz:

- ✅ Nagrywać notatki głosowe
- ✅ Uploadować wideo z automatyczną transkrypcją
- ✅ Katalogować i tagować notatki
- ✅ Wyszukiwać semantycznie
- ✅ Przeglądać chronologicznie
- ✅ Przechowywać media w chmurze

**Deploy na Streamlit Cloud**: Zobacz `DEPLOYMENT.md`
