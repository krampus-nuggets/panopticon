from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from conftest import MOCK_VECTOR
from indexer import Indexer


class TestIndexerInit:
    """Constructor correctly unpacks config values."""

    def test_defaults_from_config(self, sample_config: dict, tmp_path: Path):
        idx = Indexer(sample_config, tmp_path)

        assert idx.sources == ["src"]
        assert idx.additional_paths == []
        assert ".py" in idx.extensions
        assert idx.chunk_lines == 5
        assert idx.chunk_overlap == 2
        assert idx.table_name == "code_chunks"
        assert idx.embedding_model == "nomic-embed-text"

    def test_defaults_when_config_empty(self, tmp_path: Path):
        idx = Indexer({}, tmp_path)

        assert idx.sources == ["src"]
        assert idx.chunk_lines == 60
        assert idx.table_name == "code_chunks"


class TestResolveSources:
    """Source resolution walks directories, respects extensions/skip_dirs."""

    def test_indexes_src_directory(self, sample_config: dict, project_tree: Path):
        idx = Indexer(sample_config, project_tree)
        files = idx.resolve_sources()

        names = {f.name for f in files}
        assert "app.py" in names
        assert "utils.py" in names
        assert "notes.md" in names
        assert "deep.py" in names

    def test_skips_pycache(self, sample_config: dict, project_tree: Path):
        idx = Indexer(sample_config, project_tree)
        files = idx.resolve_sources()

        names = {f.name for f in files}
        assert "cached.py" not in names

    def test_skips_non_matching_extensions(self, sample_config: dict, project_tree: Path):
        idx = Indexer(sample_config, project_tree)
        files = idx.resolve_sources()

        names = {f.name for f in files}
        assert "data.csv" not in names

    def test_additional_paths_file(self, sample_config: dict, project_tree: Path):
        sample_config["indexer"]["additional_paths"] = ["README.md"]
        idx = Indexer(sample_config, project_tree)
        files = idx.resolve_sources()

        names = {f.name for f in files}
        assert "README.md" in names

    def test_additional_paths_directory(self, sample_config: dict, project_tree: Path):
        extra = project_tree / "docs"
        extra.mkdir()
        (extra / "guide.md").write_text("# Guide\n", encoding="utf-8")

        sample_config["indexer"]["additional_paths"] = ["docs"]
        idx = Indexer(sample_config, project_tree)
        files = idx.resolve_sources()

        names = {f.name for f in files}
        assert "guide.md" in names

    def test_missing_source_warns(self, sample_config: dict, project_tree: Path, capsys):
        sample_config["indexer"]["sources"] = ["nonexistent"]
        idx = Indexer(sample_config, project_tree)
        files = idx.resolve_sources()

        assert files == []
        assert "not found" in capsys.readouterr().out

    def test_empty_sources_returns_empty(self, sample_config: dict, project_tree: Path):
        sample_config["indexer"]["sources"] = []
        sample_config["indexer"]["additional_paths"] = []
        idx = Indexer(sample_config, project_tree)

        assert idx.resolve_sources() == []


class TestChunkFile:
    """Chunking logic splits files correctly with overlap."""

    def test_small_file_single_chunk(self, sample_config: dict, project_tree: Path):
        idx = Indexer(sample_config, project_tree)
        chunks = idx.chunk_file(project_tree / "src" / "app.py")

        assert len(chunks) == 1
        assert chunks[0]["filename"] == str(Path("src") / "app.py")
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["language"] == "python"

    def test_multi_chunk_with_overlap(self, sample_config: dict, project_tree: Path):
        """utils.py has 20 lines, chunk_lines=5, overlap=2 -> multiple chunks."""
        idx = Indexer(sample_config, project_tree)
        chunks = idx.chunk_file(project_tree / "src" / "utils.py")

        assert len(chunks) > 1
        assert chunks[0]["start_line"] == 1
        assert chunks[0]["end_line"] == 5
        assert chunks[1]["start_line"] == 4  # 5 - 2 overlap + 1

    def test_empty_file_returns_no_chunks(self, sample_config: dict, tmp_path: Path):
        empty = tmp_path / "src" / "empty.py"
        empty.parent.mkdir(parents=True, exist_ok=True)
        empty.write_text("", encoding="utf-8")

        idx = Indexer(sample_config, tmp_path)
        assert idx.chunk_file(empty) == []

    def test_language_mapping(self, sample_config: dict, project_tree: Path):
        idx = Indexer(sample_config, project_tree)
        chunks = idx.chunk_file(project_tree / "src" / "notes.md")

        assert chunks[0]["language"] == "markdown"

    def test_unknown_extension_uses_suffix(self, sample_config: dict, tmp_path: Path):
        src = tmp_path / "src"
        src.mkdir(exist_ok=True)
        exotic = src / "data.xyz"
        exotic.write_text("content\n", encoding="utf-8")

        sample_config["indexer"]["extensions"].append(".xyz")
        idx = Indexer(sample_config, tmp_path)
        chunks = idx.chunk_file(exotic)

        assert chunks[0]["language"] == "xyz"


class TestEmbedChunks:
    """_embed_chunks uses the cache and falls back to ollama."""

    @patch("ollama.embed")
    def test_cache_miss_calls_ollama(self, mock_embed, sample_config: dict, tmp_path: Path):
        mock_embed.return_value = {"embeddings": [MOCK_VECTOR]}

        idx = Indexer(sample_config, tmp_path)
        mock_cache = MagicMock()
        mock_cache.get.return_value = None

        chunks = [{"text": "hello"}]
        idx._embed_chunks(chunks, mock_cache)

        mock_embed.assert_called_once()
        mock_cache.put.assert_called_once_with("hello", MOCK_VECTOR)
        assert chunks[0]["vector"] == MOCK_VECTOR

    def test_cache_hit_skips_ollama(self, sample_config: dict, tmp_path: Path):
        idx = Indexer(sample_config, tmp_path)
        mock_cache = MagicMock()
        mock_cache.get.return_value = MOCK_VECTOR

        chunks = [{"text": "hello"}]
        with patch("ollama.embed") as mock_embed:
            idx._embed_chunks(chunks, mock_cache)

        mock_embed.assert_not_called()
        assert chunks[0]["vector"] == MOCK_VECTOR


class TestRun:
    """End-to-end run with mocked LanceDB and ollama."""

    @patch("indexer.EmbeddingCache")
    @patch("indexer.lancedb")
    @patch("ollama.embed")
    def test_run_indexes_files(
        self, mock_embed, mock_lancedb, mock_cache_cls,
        sample_config: dict, project_tree: Path,
    ):
        mock_embed.return_value = {"embeddings": [MOCK_VECTOR] * 20}
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache_cls.return_value = mock_cache_instance

        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.create_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        idx = Indexer(sample_config, project_tree)
        idx.run()

        mock_lancedb.connect.assert_called_once()
        mock_db.create_table.assert_called_once()
        mock_table.add.assert_called_once()

        added_chunks = mock_table.add.call_args[0][0]
        assert len(added_chunks) > 0
        assert "vector" in added_chunks[0]

    @patch("indexer.lancedb")
    def test_run_no_files_prints_warning(
        self, mock_lancedb, sample_config: dict, tmp_path: Path, capsys
    ):
        sample_config["indexer"]["sources"] = ["nonexistent"]
        idx = Indexer(sample_config, tmp_path)
        idx.run()

        output = capsys.readouterr().out
        assert "No files matched" in output
        mock_lancedb.connect.assert_not_called()


class TestReindexFiles:
    """Selective re-indexing of specific files."""

    @patch("indexer.EmbeddingCache")
    @patch("indexer.lancedb")
    @patch("ollama.embed")
    def test_reindex_deletes_and_inserts(
        self, mock_embed, mock_lancedb, mock_cache_cls,
        sample_config: dict, project_tree: Path,
    ):
        mock_embed.return_value = {"embeddings": [MOCK_VECTOR]}
        mock_cache_instance = MagicMock()
        mock_cache_instance.get.return_value = None
        mock_cache_cls.return_value = mock_cache_instance

        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["code_chunks"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        idx = Indexer(sample_config, project_tree)
        idx.reindex_files([project_tree / "src" / "app.py"])

        mock_table.delete.assert_called_once()
        mock_table.add.assert_called_once()

    @patch("indexer.lancedb")
    def test_reindex_no_table_prints_warning(
        self, mock_lancedb, sample_config: dict, project_tree: Path, capsys
    ):
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db

        idx = Indexer(sample_config, project_tree)
        idx.reindex_files([project_tree / "src" / "app.py"])

        assert "Run a full index first" in capsys.readouterr().out


class TestRemoveFiles:
    """Removing chunks for deleted files."""

    @patch("indexer.lancedb")
    def test_remove_deletes_from_table(
        self, mock_lancedb, sample_config: dict, project_tree: Path
    ):
        mock_table = MagicMock()
        mock_db = MagicMock()
        mock_db.table_names.return_value = ["code_chunks"]
        mock_db.open_table.return_value = mock_table
        mock_lancedb.connect.return_value = mock_db

        idx = Indexer(sample_config, project_tree)
        idx.remove_files([project_tree / "src" / "app.py"])

        mock_table.delete.assert_called_once()

    @patch("indexer.lancedb")
    def test_remove_no_table_does_nothing(
        self, mock_lancedb, sample_config: dict, project_tree: Path
    ):
        mock_db = MagicMock()
        mock_db.table_names.return_value = []
        mock_lancedb.connect.return_value = mock_db

        idx = Indexer(sample_config, project_tree)
        idx.remove_files([project_tree / "src" / "app.py"])

        mock_db.open_table.assert_not_called()
