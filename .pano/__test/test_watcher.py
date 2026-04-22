from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from watchfiles import Change

from watcher import CodeFilter, Watcher


class TestCodeFilter:
    """CodeFilter passes/rejects events based on extension and skip_dirs."""

    def setup_method(self):
        self.filter = CodeFilter(
            extensions={".py", ".md"},
            skip_dirs={"__pycache__", ".git"},
        )

    def test_accepts_matching_extension(self, tmp_path: Path):
        f = tmp_path / "src" / "app.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("x = 1\n", encoding="utf-8")
        assert self.filter(Change.modified, str(f)) is True

    def test_rejects_non_matching_extension(self, tmp_path: Path):
        f = tmp_path / "src" / "data.csv"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("a,b\n", encoding="utf-8")
        assert self.filter(Change.modified, str(f)) is False

    def test_rejects_skip_dir(self, tmp_path: Path):
        f = tmp_path / "__pycache__" / "mod.py"
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("cached = True\n", encoding="utf-8")
        assert self.filter(Change.modified, str(f)) is False

    def test_allows_deleted_events(self, tmp_path: Path):
        path = str(tmp_path / "src" / "old.py")
        assert self.filter(Change.deleted, path) is True


class TestWatcherInit:
    """Watcher constructor reads config correctly."""

    def test_debounce_from_config(self, sample_config: dict, tmp_path: Path):
        from indexer import Indexer

        indexer = Indexer(sample_config, tmp_path)
        w = Watcher(indexer, sample_config, tmp_path)

        assert w.debounce_ms == 2000

    def test_default_debounce(self, sample_config: dict, tmp_path: Path):
        from indexer import Indexer

        del sample_config["watcher"]
        indexer = Indexer(sample_config, tmp_path)
        w = Watcher(indexer, sample_config, tmp_path)

        assert w.debounce_ms == 2000


class TestWatcherResolvesPaths:
    """Watcher builds the correct list of paths to watch."""

    def test_resolves_existing_source(self, sample_config: dict, project_tree: Path):
        from indexer import Indexer

        indexer = Indexer(sample_config, project_tree)
        w = Watcher(indexer, sample_config, project_tree)
        paths = w._resolve_watch_paths()

        assert len(paths) >= 1
        assert (project_tree / "src") in paths

    def test_skips_missing_paths(self, sample_config: dict, tmp_path: Path):
        from indexer import Indexer

        sample_config["indexer"]["sources"] = ["nonexistent"]
        indexer = Indexer(sample_config, tmp_path)
        w = Watcher(indexer, sample_config, tmp_path)

        assert w._resolve_watch_paths() == []


class TestWatcherRun:
    """Watcher dispatches changes to the indexer."""

    @patch("watcher.watch")
    def test_dispatches_modified_files(
        self, mock_watch, sample_config: dict, project_tree: Path
    ):
        changed_path = str(project_tree / "src" / "app.py")
        mock_watch.return_value = iter([
            {(Change.modified, changed_path)},
        ])

        from indexer import Indexer

        indexer = Indexer(sample_config, project_tree)
        indexer.reindex_files = MagicMock()
        indexer.remove_files = MagicMock()

        w = Watcher(indexer, sample_config, project_tree)
        w.run()

        indexer.reindex_files.assert_called_once()
        call_paths = indexer.reindex_files.call_args[0][0]
        assert Path(changed_path) in call_paths
        indexer.remove_files.assert_not_called()

    @patch("watcher.watch")
    def test_dispatches_deleted_files(
        self, mock_watch, sample_config: dict, project_tree: Path
    ):
        deleted_path = str(project_tree / "src" / "old.py")
        mock_watch.return_value = iter([
            {(Change.deleted, deleted_path)},
        ])

        from indexer import Indexer

        indexer = Indexer(sample_config, project_tree)
        indexer.reindex_files = MagicMock()
        indexer.remove_files = MagicMock()

        w = Watcher(indexer, sample_config, project_tree)
        w.run()

        indexer.remove_files.assert_called_once()
        call_paths = indexer.remove_files.call_args[0][0]
        assert Path(deleted_path) in call_paths
        indexer.reindex_files.assert_not_called()

    @patch("watcher.watch")
    def test_no_valid_paths_prints_warning(
        self, mock_watch, sample_config: dict, tmp_path: Path, capsys
    ):
        sample_config["indexer"]["sources"] = ["nonexistent"]

        from indexer import Indexer

        indexer = Indexer(sample_config, tmp_path)
        w = Watcher(indexer, sample_config, tmp_path)
        w.run()

        assert "No valid paths" in capsys.readouterr().out
        mock_watch.assert_not_called()
