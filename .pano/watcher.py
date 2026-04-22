"""File system watcher for automatic incremental re-indexing."""

from pathlib import Path

from watchfiles import Change, DefaultFilter, watch

from indexer import Indexer


class CodeFilter(DefaultFilter):
    """Only pass through file events matching configured extensions and skip_dirs."""

    def __init__(self, extensions: set[str], skip_dirs: set[str]):
        self.extensions = extensions
        self.skip_dirs = skip_dirs
        super().__init__()

    def __call__(self, change: Change, path: str) -> bool:
        if not super().__call__(change, path):
            return False
        p = Path(path)
        if any(part in self.skip_dirs for part in p.parts):
            return False
        if change == Change.deleted:
            return True
        return p.suffix in self.extensions


class Watcher:
    """Watches configured source directories and triggers selective re-indexing."""

    def __init__(self, indexer: Indexer, config: dict, project_root: Path):
        self.indexer = indexer
        self.project_root = project_root

        watcher_cfg = config.get("watcher", {})
        self.debounce_ms: int = watcher_cfg.get("debounce_ms", 2000)

        self.filter = CodeFilter(indexer.extensions, indexer.skip_dirs)

    def _resolve_watch_paths(self) -> list[Path]:
        """Build the list of directories/files to watch."""
        paths: list[Path] = []
        for entry in self.indexer.sources + self.indexer.additional_paths:
            p = self.project_root / entry
            if p.exists():
                paths.append(p)
        return paths

    def run(self) -> None:
        """Block and watch for file changes, re-indexing as they occur."""
        paths = self._resolve_watch_paths()
        if not paths:
            print("No valid paths to watch.")
            return

        str_paths = [str(p) for p in paths]
        print(f"Watching {len(paths)} path(s) for changes (debounce={self.debounce_ms}ms)...")

        for changes in watch(*str_paths, watch_filter=self.filter, debounce=self.debounce_ms):
            modified: list[Path] = []
            deleted: list[Path] = []

            for change_type, path_str in changes:
                if change_type == Change.deleted:
                    deleted.append(Path(path_str))
                else:
                    modified.append(Path(path_str))

            if deleted:
                self.indexer.remove_files(deleted)
            if modified:
                self.indexer.reindex_files(modified)
