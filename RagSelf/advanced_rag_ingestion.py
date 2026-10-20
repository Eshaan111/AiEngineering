import sqlite3
import json
import hashlib
import re
import os
import ftfy

from pathlib import Path
from dataclasses import dataclass
from collections import defaultdict
from uuid import uuid5, NAMESPACE_URL
from dotenv import load_dotenv

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    Docx2txtLoader,
    TextLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from qdrant_client import QdrantClient, models


load_dotenv()


@dataclass
class IngestorConfig:
    DATA_DIR: Path = Path("data").resolve()
    STORE_DIR: Path = Path("storage")
    COLLECTION: str = "documents"
    EMBEDDING_MODEL: str = "gemini-embedding-001"
    EMBEDDING_DIM: int = 1024
    CHUNK_SIZE: int = 500
    CHUNK_OVERLAP: int = 75
    SUPPORTED: tuple = (".pdf", ".docx", ".txt")


def load_file(path: Path):
    """Load a single file using the appropriate loader."""
    ext = path.suffix.lower()

    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()

    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()

    if ext == ".txt":
        return TextLoader(
            str(path),
            encoding="utf-8",
            autodetect_encoding=True,
        ).load()

    raise ValueError(f"Unsupported file: {path}")


def normalize_text(s: str) -> str:
    s = ftfy.fix_text(s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[^\S\n]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def source_key(path: Path, data_dir: Path) -> str:
    return path.resolve().relative_to(data_dir.resolve()).as_posix()


def source_id(path: Path, data_dir: Path) -> str:
    return digest(source_key(path, data_dir).encode("utf-8"))


def file_hash(path: Path) -> str:
    return digest(path.read_bytes())


def standardized_docs(path: Path, data_dir: Path) -> list[Document]:
    """Load, normalize and standardize document metadata."""
    sid = source_id(path, data_dir)
    pieces = load_file(path)
    result = []

    for i, original_doc in enumerate(pieces):
        meta = {
            "source_id": sid,
            "source_path": source_key(path, data_dir),
            "file_type": path.suffix.lower(),
            "part_index": i,
            "page": original_doc.metadata.get("page"),
        }

        result.append(
            Document(
                page_content=normalize_text(original_doc.page_content),
                metadata=meta,
            )
        )

    return result


def connect_manifest(db_path: Path):
    conn = sqlite3.connect(db_path)

    conn.execute("""
        CREATE TABLE IF NOT EXISTS sources (
            source_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            sha256 TEXT NOT NULL,
            chunk_ids_json TEXT NOT NULL
        )
    """)

    conn.commit()
    return conn


def existing_manifest(conn):
    rows = conn.execute(
        "SELECT source_id, source_path, sha256, chunk_ids_json FROM sources"
    ).fetchall()

    return {
        r[0]: {
            "path": r[1],
            "hash": r[2],
            "ids": json.loads(r[3]),
        }
        for r in rows
    }


def chunk_source(
    path: Path,
    data_dir: Path,
    splitter,
) -> list[Document]:
    """Load and chunk a source with deterministic chunk IDs."""
    result = []

    for std_doc in standardized_docs(path, data_dir):
        chunks = splitter.split_documents([std_doc])

        for index, chunk in enumerate(chunks):
            if not chunk.page_content.strip():
                continue

            key = (
                f"{std_doc.metadata['source_id']}:"
                f"{std_doc.metadata['part_index']}:"
                f"{index}:"
                f"{chunk.page_content}"
            )

            chunk.id = digest(key.encode("utf-8"))
            chunk.metadata["chunk_index"] = index
            result.append(chunk)

    return result


def point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


def delete_chunk_ids(
    ids: list[str],
    client: QdrantClient,
    collection: str,
):
    """Delete chunks from Qdrant by their deterministic IDs."""
    if ids:
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(
                points=[point_id(i) for i in ids]
            ),
            wait=True,
        )


class Ingestor:

    def __init__(self, config: IngestorConfig):
        self.config = config

        self.config.DATA_DIR = self.config.DATA_DIR.resolve()
        self.config.STORE_DIR = self.config.STORE_DIR.resolve()

        self.config.STORE_DIR.mkdir(parents=True, exist_ok=True)

        self.QDRANT_PATH = str(self.config.STORE_DIR / "qdrant")
        self.DB_PATH = self.config.STORE_DIR / "manifest.sqlite3"

        # Discover files
        self.files = sorted(
            p for p in self.config.DATA_DIR.rglob("*")
            if p.is_file()
            and p.suffix.lower() in self.config.SUPPORTED
        ) if self.config.DATA_DIR.exists() else []

        print(f"Found {len(self.files)} supported files")

        # SQLite manifest
        self.conn = connect_manifest(self.DB_PATH)
        self.manifest = existing_manifest(self.conn)

        self.changed_files = [
            p for p in self.files
            if source_id(p, self.config.DATA_DIR) not in self.manifest
            or self.manifest[source_id(p, self.config.DATA_DIR)]["hash"]
            != file_hash(p)
        ]

        self.removed_ids = (
            set(self.manifest)
            - {source_id(p, self.config.DATA_DIR) for p in self.files}
        )

        print(
            f"New/modified: {len(self.changed_files)} | "
            f"Unchanged: {len(self.files) - len(self.changed_files)} | "
            f"Removed: {len(self.removed_ids)}"
        )

        # Chunking
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=self.config.CHUNK_SIZE,
            chunk_overlap=self.config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        self.staged = {}

        # Embeddings
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is missing from environment.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=self.config.EMBEDDING_MODEL,
            output_dimensionality=self.config.EMBEDDING_DIM,
        )

        # Qdrant
        self.client = QdrantClient(path=self.QDRANT_PATH)

        if not self.client.collection_exists(self.config.COLLECTION):
            self.client.create_collection(
                collection_name=self.config.COLLECTION,
                vectors_config=models.VectorParams(
                    size=self.config.EMBEDDING_DIM,
                    distance=models.Distance.COSINE,
                ),
            )
        else:
            existing_size = (
                self.client.get_collection(self.config.COLLECTION)
                .config.params.vectors.size
            )

            if existing_size != self.config.EMBEDDING_DIM:
                raise ValueError(
                    "Existing Qdrant collection embedding dimensions differ."
                )

        print("Qdrant collection ready:", self.config.COLLECTION)

    def ingest_one(self, path: Path):
        path = Path(path).resolve()

        if not path.is_file():
            raise FileNotFoundError(path)

        if path.suffix.lower() not in self.config.SUPPORTED:
            raise ValueError(f"Unsupported file: {path}")

        sid = source_id(path, self.config.DATA_DIR)
        current_hash = file_hash(path)

        self.manifest = existing_manifest(self.conn)

        previous = self.manifest.get(sid)

        if previous and previous["hash"] == current_hash:
            print("Unchanged, skipping:", source_key(path, self.config.DATA_DIR))
            return

        # Load and chunk
        chunks = chunk_source(
            path,
            self.config.DATA_DIR,
            self.splitter,
        )

        self.staged[sid] = chunks

        # Exact duplicate detection
        by_hash = defaultdict(list)

        for c in chunks:
            by_hash[digest(c.page_content.encode("utf-8"))].append(c.id)

        exact_groups = [
            ids for ids in by_hash.values()
            if len(ids) > 1
        ]

        print("Exact duplicate text groups:", len(exact_groups))

        previous_ids = set(previous["ids"]) if previous else set()
        new_ids = [c.id for c in chunks]

        # Embed and upload in batches
        for start in range(0, len(chunks), 64):
            batch = chunks[start:start + 64]
            texts = [c.page_content for c in batch]

            vectors = self.embeddings.embed_documents(texts)

            if len(vectors) != len(batch):
                raise ValueError("Embedding count does not match chunk count.")

            points = [
                models.PointStruct(
                    id=point_id(c.id),
                    vector=v,
                    payload={
                        "text": c.page_content,
                        **c.metadata,
                        "chunk_id": c.id,
                    },
                )
                for c, v in zip(batch, vectors)
            ]

            self.client.upsert(
                collection_name=self.config.COLLECTION,
                points=points,
                wait=True,
            )

        # Remove stale chunks
        stale_ids = list(previous_ids - set(new_ids))

        delete_chunk_ids(
            stale_ids,
            self.client,
            self.config.COLLECTION,
        )

        # Update manifest
        with self.conn:
            self.conn.execute(
                "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?)",
                (
                    sid,
                    source_key(path, self.config.DATA_DIR),
                    current_hash,
                    json.dumps(new_ids),
                ),
            )

        self.manifest = existing_manifest(self.conn)
        self.staged.pop(sid, None)

        print(
            "Indexed:",
            source_key(path, self.config.DATA_DIR),
            "chunks:",
            len(new_ids),
        )

    def remove_source(self, sid: str):
        """Remove a source and all its chunks."""
        self.manifest = existing_manifest(self.conn)

        if sid not in self.manifest:
            print("Source not found:", sid)
            return

        chunk_ids = self.manifest[sid]["ids"]

        delete_chunk_ids(
            chunk_ids,
            self.client,
            self.config.COLLECTION,
        )

        with self.conn:
            self.conn.execute(
                "DELETE FROM sources WHERE source_id = ?",
                (sid,),
            )

        self.manifest = existing_manifest(self.conn)
        self.staged.pop(sid, None)

        print("Removed source:", sid)

    def close(self):
        self.client.close()
        self.conn.close()
