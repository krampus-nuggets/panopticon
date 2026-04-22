from pathlib import Path

import lancedb

from embedding_cache import EmbeddingCache


class Indexer:
    """Indexes source files into LanceDB with embedding vectors."""

    LANG_MAP = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".json": "json",
        ".toml": "toml",
        ".md": "markdown",
        ".yaml": "yaml",
    }

    def __init__(self, config: dict, project_root: Path):
        self.config = config
        self.project_root = project_root

        idx = config.get("indexer", {})
        self.sources: list[str] = idx.get("sources", ["src"])
        self.additional_paths: list[str] = idx.get("additional_paths", [])
        self.extensions: set[str] = set(idx.get("extensions", [".py"]))
        self.skip_dirs: set[str] = set(idx.get("skip_dirs", [".venv", ".git", "__pycache__"]))
        self.chunk_lines: int = idx.get("chunk_lines", 60)
        self.chunk_overlap: int = idx.get("chunk_overlap", 10)

        db_cfg = config.get("database", {})
        self.db_path: str = str(project_root / db_cfg.get("path", "codebase-db"))
        self.table_name: str = db_cfg.get("table_name", "code_chunks")

        emb = config.get("embedding", {})
        self.embedding_model: str = emb.get("model", "nomic-embed-text")

        self._cache_path: str = str(Path(self.db_path) / ".embedding-cache")

    def resolve_sources(self) -> list[Path]:
        """Resolve configured sources and additional_paths to individual file paths."""
        files: list[Path] = []
        entries = self.sources + self.additional_paths

        for entry in entries:
            path = self.project_root / entry
            if path.is_file():
                if path.suffix in self.extensions:
                    files.append(path)
                else:
                    print(f"Warning: '{entry}' skipped (extension not in allowed list).")
            elif path.is_dir():
                for child in path.rglob("*"):
                    if any(part in self.skip_dirs for part in child.parts):
                        continue
                    if child.is_file() and child.suffix in self.extensions:
                        files.append(child)
            else:
                print(f"Warning: '{entry}' not found, skipping.")

        return files

    def chunk_file(self, path: Path) -> list[dict]:
        """Read a file and split it into overlapping line-based chunks."""
        text = path.read_text(encoding="utf-8", errors="ignore")
        lines = text.splitlines(keepends=True)
        if not lines:
            return []

        language = self.LANG_MAP.get(path.suffix, path.suffix.lstrip("."))
        rel = str(path.relative_to(self.project_root))

        chunks: list[dict] = []
        start = 0
        while start < len(lines):
            end = min(start + self.chunk_lines, len(lines))
            chunk_text = "".join(lines[start:end])
            chunks.append({
                "filename": rel,
                "start_line": start + 1,
                "end_line": end,
                "language": language,
                "text": chunk_text,
            })
            if end == len(lines):
                break
            start += self.chunk_lines - self.chunk_overlap

        return chunks

    def _get_ndims(self) -> int:
        """Determine the embedding vector dimensionality from the model."""
        import ollama

        result = ollama.embed(model=self.embedding_model, input=["test"])
        return len(result["embeddings"][0])

    def _build_schema(self, ndims: int):
        """Build a plain LanceDB schema (no auto-embedding)."""
        from lancedb.pydantic import LanceModel, Vector

        class CodeChunk(LanceModel):
            filename: str
            start_line: int
            end_line: int
            language: str
            text: str
            vector: Vector(ndims)

        return CodeChunk

    def _embed_chunks(
        self, chunks: list[dict], cache: EmbeddingCache
    ) -> list[dict]:
        """Populate each chunk with its embedding vector, using the cache."""
        import ollama

        to_embed: list[dict] = []
        for chunk in chunks:
            cached = cache.get(chunk["text"])
            if cached is not None:
                chunk["vector"] = cached
            else:
                to_embed.append(chunk)

        if to_embed:
            texts = [c["text"] for c in to_embed]
            response = ollama.embed(model=self.embedding_model, input=texts)
            for chunk, vec in zip(to_embed, response["embeddings"]):
                chunk["vector"] = vec
                cache.put(chunk["text"], vec)

        return chunks

    def run(self) -> None:
        """Execute the full indexing pipeline."""
        files = self.resolve_sources()
        if not files:
            print("No files matched the configured sources.")
            return

        all_chunks: list[dict] = []
        for f in files:
            all_chunks.extend(self.chunk_file(f))

        if not all_chunks:
            print("No chunks produced from the matched files.")
            return

        cache = EmbeddingCache(self._cache_path, self.embedding_model)
        try:
            self._embed_chunks(all_chunks, cache)
        finally:
            cache.close()

        ndims = len(all_chunks[0]["vector"])
        schema = self._build_schema(ndims)
        db = lancedb.connect(self.db_path)
        table = db.create_table(self.table_name, schema=schema, mode="overwrite")
        table.add(all_chunks)
        print(
            f"Indexed {len(all_chunks)} chunks from {len(files)} files "
            f"into '{self.table_name}' table."
        )

    def reindex_files(self, files: list[Path]) -> None:
        """Re-index specific files: delete old chunks, embed, and insert new ones."""
        rel_names = []
        all_chunks: list[dict] = []
        for f in files:
            if not f.exists():
                continue
            rel = str(f.relative_to(self.project_root))
            rel_names.append(rel)
            all_chunks.extend(self.chunk_file(f))

        db = lancedb.connect(self.db_path)
        if self.table_name not in db.table_names():
            print("Table does not exist yet. Run a full index first.")
            return

        table = db.open_table(self.table_name)

        if rel_names:
            conditions = " OR ".join(f"filename = '{name}'" for name in rel_names)
            table.delete(conditions)

        if not all_chunks:
            print(f"Removed chunks for {len(rel_names)} file(s).")
            return

        cache = EmbeddingCache(self._cache_path, self.embedding_model)
        try:
            self._embed_chunks(all_chunks, cache)
        finally:
            cache.close()

        table.add(all_chunks)
        print(
            f"Re-indexed {len(all_chunks)} chunks from {len(rel_names)} file(s)."
        )

    def remove_files(self, files: list[Path]) -> None:
        """Remove all chunks for the given files from the table."""
        db = lancedb.connect(self.db_path)
        if self.table_name not in db.table_names():
            return

        table = db.open_table(self.table_name)
        rel_names = []
        for f in files:
            try:
                rel_names.append(str(f.relative_to(self.project_root)))
            except ValueError:
                continue

        if rel_names:
            conditions = " OR ".join(f"filename = '{name}'" for name in rel_names)
            table.delete(conditions)
            print(f"Removed chunks for {len(rel_names)} deleted file(s).")
