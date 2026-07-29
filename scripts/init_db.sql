-- Enable pgvector and create the schema.
-- Runs automatically the first time the db container starts.
-- If you change this after first boot: docker compose down -v && make db-up
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id            BIGSERIAL PRIMARY KEY,
    source        TEXT NOT NULL,          -- path / stable id the chunk came from
    chunk_index   INT  NOT NULL,          -- position of chunk within source
    content       TEXT NOT NULL,          -- the chunk text
    content_hash  TEXT NOT NULL,          -- sha256 of content (idempotent upserts)
    metadata      JSONB NOT NULL DEFAULT '{}'::jsonb,
    embedding     VECTOR(768) NOT NULL,  -- must match EMBEDDING_DIM
    -- Full-text vector for hybrid retrieval (updated on insert/upsert).
    search_vector tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    UNIQUE (source, chunk_index)
);

CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);

CREATE INDEX IF NOT EXISTS documents_search_vector_idx
    ON documents USING gin (search_vector);

CREATE INDEX IF NOT EXISTS documents_metadata_idx
    ON documents USING gin (metadata);

CREATE INDEX IF NOT EXISTS documents_source_idx
    ON documents (source);
