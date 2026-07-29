-- Enable pgvector and create the schema.
-- Runs automatically the first time the db container starts.
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS documents (
    id          BIGSERIAL PRIMARY KEY,
    source      TEXT NOT NULL,          -- filename / URL the chunk came from
    chunk_index INT  NOT NULL,          -- position of chunk within source
    content     TEXT NOT NULL,          -- the chunk text
    metadata    JSONB DEFAULT '{}',     -- date, author, category, etc.
    embedding   VECTOR(768)             -- must match EMBEDDING_DIM (Ollama nomic-embed-text / bge-base)
);

-- Approximate-nearest-neighbour index for fast similarity search.
-- Cosine distance; swap to vector_l2_ops if you prefer L2.
CREATE INDEX IF NOT EXISTS documents_embedding_idx
    ON documents USING hnsw (embedding vector_cosine_ops);
