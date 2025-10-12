-- Inicjalizacja bazy danych dla aplikacji Audio & Video Notes

-- Tabela z notatkami
CREATE TABLE IF NOT EXISTS notes (
    id SERIAL PRIMARY KEY,
    text TEXT NOT NULL,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    source_type VARCHAR(20) DEFAULT 'audio' CHECK (source_type IN ('audio', 'video', 'text')),
    media_url TEXT,
    media_type VARCHAR(50),
    qdrant_id INTEGER,  -- ID punktu w Qdrant
    timestamps JSONB,  -- Timestampy segmentów z transkrypcji (dla audio/wideo)
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela z kategoriami
CREATE TABLE IF NOT EXISTS categories (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Tabela powiązań notatka-kategoria (many-to-many)
CREATE TABLE IF NOT EXISTS note_categories (
    note_id INTEGER REFERENCES notes(id) ON DELETE CASCADE,
    category_id INTEGER REFERENCES categories(id) ON DELETE CASCADE,
    PRIMARY KEY (note_id, category_id)
);

-- Indeksy dla lepszej wydajności
CREATE INDEX IF NOT EXISTS idx_notes_timestamp ON notes(timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_notes_source_type ON notes(source_type);
CREATE INDEX IF NOT EXISTS idx_notes_qdrant_id ON notes(qdrant_id);
CREATE INDEX IF NOT EXISTS idx_categories_name ON categories(name);

-- Funkcja do automatycznej aktualizacji updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Trigger dla automatycznej aktualizacji updated_at
DROP TRIGGER IF EXISTS update_notes_updated_at ON notes;
CREATE TRIGGER update_notes_updated_at
    BEFORE UPDATE ON notes
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Widok do łatwego pobierania notatek z kategoriami
CREATE OR REPLACE VIEW notes_with_categories AS
SELECT 
    n.id,
    n.text,
    n.timestamp,
    n.source_type,
    n.media_url,
    n.media_type,
    n.qdrant_id,
    n.created_at,
    n.updated_at,
    COALESCE(
        json_agg(
            json_build_object('id', c.id, 'name', c.name)
        ) FILTER (WHERE c.id IS NOT NULL),
        '[]'::json
    ) as categories
FROM notes n
LEFT JOIN note_categories nc ON n.id = nc.note_id
LEFT JOIN categories c ON nc.category_id = c.id
GROUP BY n.id;

-- Przykładowe dane testowe (opcjonalne - możesz zakomentować)
-- INSERT INTO categories (name) VALUES ('Praca'), ('Osobiste'), ('Projekt');

