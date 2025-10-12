# PostgreSQL Setup - Instrukcje

## Architektura

Aplikacja wykorzystuje **hybrydowe podejście**:

- **PostgreSQL** - główna baza danych do przechowywania metadanych notatek, kategorii i relacji
- **Qdrant** - baza wektorowa wyłącznie do semantycznego wyszukiwania
- **DigitalOcean Spaces** - przechowywanie plików audio/wideo

## Zalety PostgreSQL

✅ **Relacyjne dane** - kategorie, notatki, powiązania many-to-many
✅ **SQL queries** - zaawansowane filtrowanie i analiza
✅ **Transakcje** - integralność danych (ACID)
✅ **Backup** - łatwe kopie zapasowe
✅ **Indeksy** - szybkie wyszukiwanie po dacie, kategorii
✅ **Audyt** - timestamps (created_at, updated_at)

## Konfiguracja DigitalOcean PostgreSQL

### 1. Utworzenie bazy danych

1. Zaloguj się do [DigitalOcean Cloud](https://cloud.digitalocean.com/)
2. Przejdź do **Databases** → **Create Database**
3. Wybierz:
   - **Database Engine**: PostgreSQL (najnowsza wersja)
   - **Data Center Region**: wybierz region najbliżej Ciebie (np. Frankfurt)
   - **Plan**: Basic ($15/mo dla startu)
4. Kliknij **Create Database Cluster**

### 2. Konfiguracja połączenia

Po utworzeniu bazy, w zakładce **Overview** znajdziesz dane połączenia:

```
Host: your-db-host.db.ondigitalocean.com
Port: 25060
Database: defaultdb
User: doadmin
Password: ************
SSL Mode: require
```

### 3. Dodaj dane do pliku .env

Skopiuj `env_example.txt` do `.env` i uzupełnij:

```bash
# PostgreSQL Database (DigitalOcean)
POSTGRES_HOST=your-db-host.db.ondigitalocean.com
POSTGRES_PORT=25060
POSTGRES_DB=defaultdb
POSTGRES_USER=doadmin
POSTGRES_PASSWORD=twoje-haslo-z-do
POSTGRES_SSLMODE=require
```

### 4. Inicjalizacja tabel

Aplikacja **automatycznie** utworzy tabele przy pierwszym uruchomieniu używając skryptu `init_db.sql`.

Jeśli chcesz ręcznie zainicjalizować bazę:

```bash
# Połącz się z bazą (z poziomu terminala DigitalOcean lub lokalnie)
psql "postgresql://doadmin:haslo@host.db.ondigitalocean.com:25060/defaultdb?sslmode=require"

# Następnie wykonaj
\i init_db.sql
```

## Struktura bazy danych

### Tabela: notes

Główna tabela z notatkami:

- `id` - unikalny identyfikator
- `text` - treść notatki
- `timestamp` - data utworzenia
- `source_type` - typ źródła (audio/video/text)
- `media_url` - link do pliku w Spaces
- `media_type` - MIME type pliku
- `qdrant_id` - ID punktu w bazie wektorowej
- `created_at`, `updated_at` - timestamps

### Tabela: categories

Kategorie notatek:

- `id` - unikalny identyfikator
- `name` - nazwa kategorii (UNIQUE)
- `created_at` - data utworzenia

### Tabela: note_categories

Relacja many-to-many między notatkami a kategoriami:

- `note_id` - FK do notes
- `category_id` - FK do categories

### Widok: notes_with_categories

Ułatwia pobieranie notatek z kategoriami w jednym zapytaniu.

## Jak działa wyszukiwanie

### Przeglądanie (bez query)

1. Dane pobierane **tylko z PostgreSQL**
2. Szybkie filtrowanie SQL po kategorii, dacie
3. Sortowanie po `timestamp`

### Wyszukiwanie semantyczne (z query)

1. Query → embedding (OpenAI)
2. **Qdrant** - wyszukanie podobnych wektorów
3. **PostgreSQL** - pobranie pełnych danych po ID
4. Sortowanie wg `score` z Qdrant

## Zarządzanie bazą

### Backup

```bash
# Utwórz backup
pg_dump "postgresql://doadmin:haslo@host:25060/defaultdb?sslmode=require" > backup.sql

# Przywróć z backupu
psql "postgresql://doadmin:haslo@host:25060/defaultdb?sslmode=require" < backup.sql
```

### Przydatne zapytania SQL

```sql
-- Statystyki notatek
SELECT
    source_type,
    COUNT(*) as count,
    MIN(timestamp) as oldest,
    MAX(timestamp) as newest
FROM notes
GROUP BY source_type;

-- Top 10 kategorii
SELECT
    c.name,
    COUNT(nc.note_id) as note_count
FROM categories c
LEFT JOIN note_categories nc ON c.id = nc.category_id
GROUP BY c.name
ORDER BY note_count DESC
LIMIT 10;

-- Notatki bez kategorii
SELECT id, text, timestamp
FROM notes n
WHERE NOT EXISTS (
    SELECT 1 FROM note_categories nc WHERE nc.note_id = n.id
);
```

## Migracja z samego Qdrant

Jeśli wcześniej używałeś tylko Qdrant, aplikacja automatycznie obsłuży backward compatibility:

- Brak PostgreSQL → działa jak wcześniej (tylko Qdrant)
- PostgreSQL skonfigurowany → nowe notatki w PostgreSQL + Qdrant

Stare notatki pozostaną w Qdrant i będą działać.

## Troubleshooting

### Błąd: "could not connect to server"

- Sprawdź czy IP jest dozwolone w DigitalOcean (Databases → Settings → Trusted Sources)
- Dodaj swoje IP lub `0.0.0.0/0` dla wszystkich

### Błąd: "SSL connection required"

- Upewnij się, że `POSTGRES_SSLMODE=require` w `.env`

### Błąd podczas init_db.sql

- Sprawdź czy użytkownik ma uprawnienia CREATE TABLE
- Użytkownik `doadmin` ma pełne uprawnienia domyślnie

## Koszt

- PostgreSQL Basic (1 GB RAM, 10 GB storage): **$15/miesiąc**
- Qdrant Cloud: **Free tier** (1 GB)
- DigitalOcean Spaces: **$5/miesiąc** (250 GB transfer)

**Razem:** ~$20/miesiąc dla pełnego stacku produkcyjnego.

