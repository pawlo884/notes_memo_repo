# 🔐 Zabezpieczenie dostępu do aplikacji

Aplikacja posiada wbudowany system logowania, który pozwala ograniczyć dostęp tylko do autoryzowanych użytkowników.

## Włączenie logowania

### Krok 1: Wygeneruj hash hasła

Uruchom skrypt pomocniczy:

```bash
python generate_password_hash.py
```

Skrypt zapyta Cię o hasło i wygeneruje hash SHA256.

### Krok 2: Dodaj credentiale do `.env`

Dodaj poniższe linie do pliku `.env`:

```env
# System logowania
APP_USERNAME=twoj_login
APP_PASSWORD_HASH=wygenerowany_hash_z_kroku_1
```

### Przykład:

```env
APP_USERNAME=admin
APP_PASSWORD_HASH=8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918
```

⚠️ **UWAGA**: Powyższy hash to hasło `admin` - **ZMIEŃ TO** dla bezpieczeństwa!

### Krok 3: Uruchom aplikację

Po dodaniu zmiennych do `.env`, aplikacja automatycznie wyświetli ekran logowania.

```bash
streamlit run app.py
```

## Wyłączenie logowania

Jeśli chcesz wyłączyć system logowania, po prostu **usuń** lub **zakomentuj** zmienne `APP_USERNAME` i `APP_PASSWORD_HASH` z pliku `.env`:

```env
# APP_USERNAME=admin
# APP_PASSWORD_HASH=8c6976e5b5410415bde908bd4dee15dfb167a9c873fc4bb8a81f6f2ab448a918
```

## Funkcje

✅ **Formularz logowania** - pojawia się automatycznie gdy są ustawione credentiale  
✅ **Hashowanie SHA256** - hasła są bezpiecznie hashowane  
✅ **Session state** - użytkownik pozostaje zalogowany przez całą sesję  
✅ **Przycisk wylogowania** - w sidebarze, umożliwia łatwe wylogowanie  
✅ **Opcjonalne** - jeśli nie ustawisz credentiali, aplikacja działa bez logowania

## Generowanie hasha ręcznie (opcjonalnie)

Jeśli wolisz wygenerować hash hasła ręcznie w Pythonie:

```python
from hashlib import sha256

password = "twoje_haslo"
password_hash = sha256(password.encode()).hexdigest()
print(password_hash)
```

## Bezpieczeństwo

- ✅ Hasła są hashowane SHA256 (nigdy nie są przechowywane jako plaintext)
- ✅ Hasła nie są widoczne w kodzie źródłowym (tylko w `.env`)
- ✅ `.env` jest w `.gitignore` (nie trafia do repozytorium)
- ⚠️ **Używaj silnych haseł** (minimum 8 znaków, małe i duże litery, cyfry, znaki specjalne)
- ⚠️ Dla publicznych aplikacji rozważ dodatkowe zabezpieczenia (HTTPS, rate limiting, 2FA)

## Dodatkowe zabezpieczenia (opcjonalne)

Dla jeszcze większego bezpieczeństwa, możesz:

1. **Używać HTTPS** - szczególnie w deployment (np. Streamlit Cloud)
2. **Ograniczenie IP** - zezwalaj na dostęp tylko z określonych adresów IP (konfiguracja serwera/proxy)
3. **Rate limiting** - ogranicz liczbę prób logowania
4. **2FA** - dwuskładnikowa autentykacja (wymaga dodatkowej biblioteki)

## Konfiguracja lokalnie (opcja 1: .env)

Dodaj zmienne do pliku `.env`:

```env
APP_USERNAME=admin
APP_PASSWORD_HASH=twoj_hash
OPENAI_API_KEY=sk-your-key
```

## Konfiguracja lokalnie (opcja 2: secrets.toml)

Utwórz plik `.streamlit/secrets.toml`:

```bash
mkdir -p .streamlit
```

Dodaj zawartość:

```toml
APP_USERNAME = "admin"
APP_PASSWORD_HASH = "twoj_hash"
OPENAI_API_KEY = "sk-your-key"
```

Zobacz przykładowy plik: `secrets.toml.example`

## Deployment na Streamlit Cloud

Jeśli deployujesz na Streamlit Cloud, dodaj zmienne w **Secrets management**:

1. Przejdź do: Settings → Secrets
2. Wklej całą zawartość z `secrets.toml.example` i wypełnij własnymi wartościami:

```toml
OPENAI_API_KEY = "sk-your-key"
APP_USERNAME = "twoj_login"
APP_PASSWORD_HASH = "twoj_hash"
# ... reszta zmiennych
```

Aplikacja automatycznie użyje secrets z Streamlit Cloud jeśli są dostępne.
