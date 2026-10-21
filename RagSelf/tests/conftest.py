"""Test fixtures: isolated SQLite + local Qdrant, no Gemini API calls."""
import hashlib
import math

import pytest

import ingestor as module


class FakeEmbeddings:
    """Stable, normalized vectors derived from text; not semantic embeddings."""

    instances = []

    def __init__(self, model, output_dimensionality):
        self.dim = output_dimensionality
        self.calls = []
        self.instances.append(self)

    def embed_documents(self, texts):
        self.calls.append(list(texts))
        out = []
        for text in texts:
            # Repeat SHA-256 bytes to support any small test dimensionality.
            seed = hashlib.sha256(text.encode('utf-8')).digest()
            values = [(seed[i % len(seed)] - 127.5) / 127.5 for i in range(self.dim)]
            norm = math.sqrt(sum(x * x for x in values)) or 1.0
            out.append([x / norm for x in values])
        return out


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    monkeypatch.setenv('GOOGLE_API_KEY', 'unit-test-only-no-network')
    monkeypatch.setattr(module, 'GoogleGenerativeAIEmbeddings', FakeEmbeddings)
    data = tmp_path / 'data'
    store = tmp_path / 'storage'
    data.mkdir()
    config = module.IngestorConfig(
        DATA_DIR=data, STORE_DIR=store, COLLECTION='tests',
        EMBEDDING_DIM=8, CHUNK_SIZE=80, CHUNK_OVERLAP=10,
        BATCH_SIZE=4, STRUCTURE_AWARE=True,
    )
    engine = module.Ingestor(config)
    try:
        yield engine, data
    finally:
        engine.close()


def write_txt(data, name, text):
    path = data / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding='utf-8')
    return path


@pytest.fixture
def sample_text():
    return 'Leave policy: Each employee gets twenty days annually.\n\n' * 6
