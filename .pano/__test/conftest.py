import sys
from pathlib import Path

import pytest

PANO_DIR = Path(__file__).resolve().parent.parent
if str(PANO_DIR) not in sys.path:
    sys.path.insert(0, str(PANO_DIR))


MOCK_VECTOR = [0.1] * 768


@pytest.fixture()
def sample_config() -> dict:
    """Minimal valid pano.yaml config as a dict."""
    return {
        "indexer": {
            "sources": ["src"],
            "additional_paths": [],
            "extensions": [".py", ".md"],
            "skip_dirs": [".venv", ".git", "__pycache__"],
            "chunk_lines": 5,
            "chunk_overlap": 2,
        },
        "database": {
            "path": "codebase-db",
            "table_name": "code_chunks",
        },
        "embedding": {
            "provider": "ollama",
            "model": "nomic-embed-text",
        },
        "watcher": {
            "debounce_ms": 2000,
        },
        "mcp": {
            "server_name": "pano-test",
            "transport": "stdio",
        },
    }


@pytest.fixture()
def project_tree(tmp_path: Path) -> Path:
    """Create a minimal project tree with source files for indexing tests."""
    src = tmp_path / "src"
    src.mkdir()

    (src / "app.py").write_text(
        "def hello():\n    return 'world'\n",
        encoding="utf-8",
    )
    (src / "utils.py").write_text(
        "\n".join(f"line {i}" for i in range(1, 21)) + "\n",
        encoding="utf-8",
    )
    (src / "notes.md").write_text("# Notes\nSome content.\n", encoding="utf-8")
    (src / "data.csv").write_text("a,b\n1,2\n", encoding="utf-8")

    nested = src / "sub"
    nested.mkdir()
    (nested / "deep.py").write_text("x = 1\n", encoding="utf-8")

    skip = src / "__pycache__"
    skip.mkdir()
    (skip / "cached.py").write_text("cached = True\n", encoding="utf-8")

    (tmp_path / "README.md").write_text("# README\n", encoding="utf-8")

    return tmp_path


@pytest.fixture()
def config_yaml_file(tmp_path: Path, sample_config: dict) -> Path:
    """Write sample_config to a pano.yaml file and return its path."""
    import yaml

    cfg_path = tmp_path / "pano.yaml"
    cfg_path.write_text(yaml.dump(sample_config), encoding="utf-8")
    return cfg_path
