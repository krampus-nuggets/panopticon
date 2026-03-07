from pathlib import Path

import lancedb


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

    def _build_schema(self):
        """Build the LanceDB schema with the configured embedding model.

        Deferred to avoid requiring Ollama at import time.
        """
        from lancedb.embeddings import get_registry
        from lancedb.pydantic import LanceModel, Vector

        func = get_registry().get("ollama").create(name=self.embedding_model)

        class CodeChunk(LanceModel):
            filename: str
            start_line: int
            end_line: int
            language: str
            text: str = func.SourceField()
            vector: Vector(func.ndims()) = func.VectorField()

        return CodeChunk

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

        schema = self._build_schema()
        db = lancedb.connect(self.db_path)
        table = db.create_table(self.table_name, schema=schema, mode="overwrite")
        table.add(all_chunks)
        print(
            f"Indexed {len(all_chunks)} chunks from {len(files)} files "
            f"into '{self.table_name}' table."
        )
