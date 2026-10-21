"""Reusable PDF/DOCX/TXT ingestion into a local Qdrant collection.

Optional near-duplicate reporting requires: pip install rapidfuzz
"""
import hashlib
import json
import os
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TypedDict
from uuid import NAMESPACE_URL, uuid5

import ftfy
from dotenv import load_dotenv
from langchain_community.document_loaders import Docx2txtLoader, PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
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
    STRUCTURE_AWARE: bool = True
    BATCH_SIZE: int = 64


class ChunkPayload(TypedDict):
    text: str
    source_id: str
    source_path: str
    file_type: str
    part_index: int
    page: int | None
    chunk_index: int
    chunk_id: str
    content_hash: str


def load_file(path: Path):
    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext == ".docx":
        return Docx2txtLoader(str(path)).load()
    if ext == ".txt":
        return TextLoader(str(path), encoding="utf-8", autodetect_encoding=True).load()
    raise ValueError(f"Unsupported file: {path}")


def normalize_text(s: str) -> str:
    s = ftfy.fix_text(s).replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"[^\S\n]+", " ", s)
    s = re.sub(r" *\n *", "\n", s)
    return re.sub(r"\n{3,}", "\n\n", s).strip()


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def content_hash(text: str) -> str:
    return digest(text.encode("utf-8"))


def source_key(path: Path, data_dir: Path) -> str:
    return Path(path).resolve().relative_to(Path(data_dir).resolve()).as_posix()


def source_id(path: Path, data_dir: Path) -> str:
    return digest(source_key(path, data_dir).encode("utf-8"))


def file_hash(path: Path) -> str:
    return digest(Path(path).read_bytes())


def standardized_docs(path: Path, data_dir: Path) -> list[Document]:
    sid = source_id(path, data_dir)
    result = []
    for i, original in enumerate(load_file(path)):
        result.append(Document(
            page_content=normalize_text(original.page_content),
            metadata={
                "source_id": sid,
                "source_path": source_key(path, data_dir),
                "file_type": path.suffix.lower(),
                "part_index": i,
                "page": original.metadata.get("page"),
            },
        ))
    return result


def make_payload(chunk: Document) -> ChunkPayload:
    m = chunk.metadata
    return {
        "text": chunk.page_content,
        "source_id": m["source_id"],
        "source_path": m["source_path"],
        "file_type": m["file_type"],
        "part_index": m["part_index"],
        "page": m["page"],
        "chunk_index": m["chunk_index"],
        "chunk_id": chunk.id,
        "content_hash": content_hash(chunk.page_content),
    }


def connect_manifest(db_path: Path):
    conn = sqlite3.connect(db_path)
    conn.execute("""CREATE TABLE IF NOT EXISTS sources (
        source_id TEXT PRIMARY KEY,
        source_path TEXT NOT NULL,
        sha256 TEXT NOT NULL,
        chunk_ids_json TEXT NOT NULL
    )""")
    conn.commit()
    return conn


def existing_manifest(conn):
    rows = conn.execute(
        "SELECT source_id, source_path, sha256, chunk_ids_json FROM sources"
    ).fetchall()
    return {r[0]: {"path": r[1], "hash": r[2], "ids": json.loads(r[3])} for r in rows}


def _sections(doc: Document, structure_aware: bool) -> list[Document]:
    if not structure_aware:
        return [doc]
    # This is heading-aware for Markdown-like text; it does not infer headings
    # or tables from PDF page geometry or DOCX styles.
    headings = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    ).split_text(doc.page_content)
    if not headings:
        return [doc]
    return [Document(page_content=s.page_content, metadata=doc.metadata.copy()) for s in headings]


def chunk_source(path: Path, data_dir: Path, splitter,
                 structure_aware: bool = False) -> list[Document]:
    result = []
    for std_doc in standardized_docs(path, data_dir):
        index = 0
        for section in _sections(std_doc, structure_aware):
            for chunk in splitter.split_documents([section]):
                if not chunk.page_content.strip():
                    continue
                key = (f"{std_doc.metadata['source_id']}:"
                       f"{std_doc.metadata['part_index']}:{index}:{chunk.page_content}")
                chunk.id = digest(key.encode("utf-8"))
                chunk.metadata["chunk_index"] = index
                result.append(chunk)
                index += 1
    return result


def point_id(chunk_id: str) -> str:
    return str(uuid5(NAMESPACE_URL, chunk_id))


def delete_chunk_ids(ids: list[str], client: QdrantClient, collection: str):
    if ids:
        client.delete(
            collection_name=collection,
            points_selector=models.PointIdsList(points=[point_id(i) for i in ids]),
            wait=True,
        )


class Ingestor:
    def __init__(self, config: IngestorConfig):
        self.config = config
        self.config.DATA_DIR = Path(config.DATA_DIR).resolve()
        self.config.STORE_DIR = Path(config.STORE_DIR).resolve()
        self.config.STORE_DIR.mkdir(parents=True, exist_ok=True)
        self.QDRANT_PATH = str(self.config.STORE_DIR / "qdrant")
        self.DB_PATH = self.config.STORE_DIR / "manifest.sqlite3"
        self.conn = connect_manifest(self.DB_PATH)
        self.manifest = existing_manifest(self.conn)
        self.splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            encoding_name="cl100k_base",
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        self.staged = {}
        self.scan_changes()
        if not os.getenv("GOOGLE_API_KEY"):
            raise ValueError("GOOGLE_API_KEY is missing from environment")
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            output_dimensionality=config.EMBEDDING_DIM,
        )
        self.client = QdrantClient(path=self.QDRANT_PATH)
        if not self.client.collection_exists(config.COLLECTION):
            self.client.create_collection(
                collection_name=config.COLLECTION,
                vectors_config=models.VectorParams(
                    size=config.EMBEDDING_DIM, distance=models.Distance.COSINE,
                ),
            )
        else:
            size = self.client.get_collection(config.COLLECTION).config.params.vectors.size
            if size != config.EMBEDDING_DIM:
                raise ValueError("Existing Qdrant collection vector dimensions differ")
        print("Qdrant collection ready:", config.COLLECTION)

    def scan_changes(self):
        """Refresh changed_files/removed_ids after filesystem changes."""
        self.manifest = existing_manifest(self.conn)
        root = self.config.DATA_DIR
        self.files = sorted(
            p for p in root.rglob("*")
            if p.is_file() and p.suffix.lower() in self.config.SUPPORTED
        ) if root.exists() else []
        ids = {source_id(p, root) for p in self.files}
        self.changed_files = [
            p for p in self.files
            if (sid := source_id(p, root)) not in self.manifest
            or self.manifest[sid]["hash"] != file_hash(p)
        ]
        self.removed_ids = set(self.manifest) - ids
        print(f"New/modified: {len(self.changed_files)} | "
              f"Unchanged: {len(self.files) - len(self.changed_files)} | "
              f"Removed: {len(self.removed_ids)}")
        return self.changed_files, self.removed_ids

    def stored_chunks(self):
        """Stream Qdrant points using scroll pagination (without vectors)."""
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.config.COLLECTION, limit=256,
                offset=offset, with_payload=True, with_vectors=False,
            )
            yield from points
            if offset is None:
                break

    def exact_duplicate_report(self, incoming: list[Document]):
        """Report same-text chunks in Qdrant and incoming batch; no deletion."""
        grouped = defaultdict(list)
        for p in self.stored_chunks():
            meta = p.payload or {}
            if meta.get("text"):
                grouped[content_hash(meta["text"])].append(
                    ("stored", meta.get("source_path"), str(p.id)))
        for c in incoming:
            if c.page_content:
                grouped[content_hash(c.page_content)].append(
                    ("incoming", c.metadata.get("source_path"), c.id))
        return {h: refs for h, refs in grouped.items() if len(refs) > 1}

    def near_duplicate_report(self, incoming: list[Document], threshold: float = 90.0,
                              max_existing: int = 200, max_incoming: int = 50):
        """Bounded textual comparison, not semantic deduplication."""
        from rapidfuzz.fuzz import ratio
        existing = []
        for p in self.stored_chunks():
            t = (p.payload or {}).get("text", "")
            if t:
                existing.append((str(p.id), t))
            if len(existing) >= max_existing:
                break
        hits = []
        for c in incoming[:max_incoming]:
            for old_id, old_text in existing:
                if c.page_content == old_text:
                    continue
                score = ratio(c.page_content, old_text)
                if score >= threshold:
                    hits.append((round(score, 1), c.id, old_id))
        return sorted(hits, reverse=True)

    def ingest_one(self, path: Path):
        """Incremental ingest; reuses vectors already present for the same content."""
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
            return {"skipped": True, "reused": 0, "embedded": 0}

        chunks = chunk_source(path, self.config.DATA_DIR, self.splitter,
                              structure_aware=self.config.STRUCTURE_AWARE)
        self.staged[sid] = chunks
        previous_ids = set(previous["ids"]) if previous else set()
        new_ids = {c.id for c in chunks}
        # Map identical content to an already existing vector, even across sources.
        existing_hash_points = {}
        for p in self.stored_chunks():
            m = p.payload or {}
            text = m.get("text", "")
            if text:
                existing_hash_points.setdefault(content_hash(text), p.id)

        reused = embedded = 0
        batch_size = self.config.BATCH_SIZE
        try:
            for start in range(0, len(chunks), batch_size):
                batch = chunks[start:start + batch_size]
                desired_ids = [point_id(c.id) for c in batch]
                # Existing point IDs may already be present after an interrupted run.
                old_points = self.client.retrieve(
                    collection_name=self.config.COLLECTION, ids=desired_ids,
                    with_vectors=True, with_payload=False,
                )
                vectors = {str(p.id): p.vector for p in old_points}
                # For exact duplicates, retrieve the existing vector once.
                candidates = {}
                for c in batch:
                    if point_id(c.id) in vectors:
                        continue
                    h = content_hash(c.page_content)
                    if h in existing_hash_points:
                        candidates[h] = existing_hash_points[h]
                if candidates:
                    retrieved = self.client.retrieve(
                        collection_name=self.config.COLLECTION,
                        ids=list(set(candidates.values())),
                        with_vectors=True, with_payload=False,
                    )
                    by_point = {str(p.id): p.vector for p in retrieved}
                    for c in batch:
                        pid = point_id(c.id)
                        old_pid = candidates.get(content_hash(c.page_content))
                        if pid not in vectors and old_pid is not None:
                            vector = by_point.get(str(old_pid))
                            if isinstance(vector, list):
                                vectors[pid] = vector

                # Avoid embedding identical text twice within one batch.
                pending = {}
                for c in batch:
                    pid = point_id(c.id)
                    if pid not in vectors:
                        pending.setdefault(content_hash(c.page_content), c.page_content)
                hashes = list(pending)
                new_vectors = self.embeddings.embed_documents([pending[h] for h in hashes]) if hashes else []
                if len(new_vectors) != len(hashes):
                    raise ValueError("Embedding count does not match requested texts")
                generated = dict(zip(hashes, new_vectors))
                for c in batch:
                    pid = point_id(c.id)
                    if pid in vectors:
                        reused += 1
                    else:
                        vectors[pid] = generated[content_hash(c.page_content)]
                        # Counts chunks assigned a new embedding; repeated texts in
                        # this batch may share one API embedding request.
                        embedded += 1
                points = [models.PointStruct(
                    id=point_id(c.id), vector=vectors[point_id(c.id)],
                    payload=make_payload(c),
                ) for c in batch]
                if points:
                    self.client.upsert(
                        collection_name=self.config.COLLECTION,
                        points=points, wait=True,
                    )
                for c in batch:
                    existing_hash_points[content_hash(c.page_content)] = point_id(c.id)

            # Only delete old chunks after all new ones were uploaded.
            delete_chunk_ids(list(previous_ids - new_ids), self.client, self.config.COLLECTION)
            with self.conn:
                self.conn.execute(
                    "INSERT OR REPLACE INTO sources VALUES (?, ?, ?, ?)",
                    (sid, source_key(path, self.config.DATA_DIR), current_hash,
                     json.dumps([c.id for c in chunks])),
                )
            self.manifest = existing_manifest(self.conn)
            print(f"Indexed: {source_key(path, self.config.DATA_DIR)} | "
                  f"chunks: {len(chunks)} | reused: {reused} | embedded: {embedded}")
            return {"skipped": False, "chunks": len(chunks), "reused": reused,
                    "embedded": embedded}
        finally:
            self.staged.pop(sid, None)

    def remove_source(self, sid: str):
        self.manifest = existing_manifest(self.conn)
        if sid not in self.manifest:
            print("Source not found:", sid)
            return
        delete_chunk_ids(self.manifest[sid]["ids"], self.client, self.config.COLLECTION)
        with self.conn:
            self.conn.execute("DELETE FROM sources WHERE source_id = ?", (sid,))
        self.manifest = existing_manifest(self.conn)
        self.staged.pop(sid, None)
        print("Removed source:", sid)

    def validate_ingestion(self):
        """Read-only manifest/Qdrant consistency report, also flags source changes."""
        manifest = existing_manifest(self.conn)
        expected = {point_id(cid) for item in manifest.values() for cid in item["ids"]}
        actual = {str(p.id) for p in self.stored_chunks()}
        missing_files, modified_files = [], []
        for item in manifest.values():
            path = self.config.DATA_DIR / item["path"]
            if not path.is_file():
                missing_files.append(item["path"])
            elif file_hash(path) != item["hash"]:
                modified_files.append(item["path"])
        return {
            "manifest_sources": len(manifest),
            "manifest_points": len(expected),
            "qdrant_points": len(actual),
            "missing_in_qdrant": sorted(expected - actual),
            "orphaned_in_qdrant": sorted(actual - expected),
            "missing_source_files": missing_files,
            "modified_source_files": modified_files,
        }

    def cleanup_orphans(self, dry_run: bool = True):
        """Delete unreferenced points ONLY on explicit dry_run=False."""
        ids = self.validate_ingestion()["orphaned_in_qdrant"]
        if ids and not dry_run:
            self.client.delete(
                collection_name=self.config.COLLECTION,
                points_selector=models.PointIdsList(points=ids), wait=True,
            )
        return ids

    def close(self):
        self.client.close()
        self.conn.close()
