import sys
import threading
from pathlib import Path
from typing import Optional

import typer
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / ".pano"))

app = typer.Typer(name="pano", help="Panopticon -- local codebase indexing & RAG tools.")


def load_config(config_path: Optional[Path] = None) -> dict:
    """Load and return the YAML configuration."""
    path = config_path or PROJECT_ROOT / "pano.yaml"
    if not path.exists():
        typer.echo(f"Error: config file not found at '{path}'", err=True)
        raise typer.Exit(code=1)

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        typer.echo(f"Error: config file '{path}' is empty or malformed", err=True)
        raise typer.Exit(code=1)

    return data


@app.command()
def index(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to pano.yaml config file."),
) -> None:
    """Index the codebase into LanceDB for semantic search."""
    from indexer import Indexer

    cfg = load_config(config)
    indexer = Indexer(cfg, PROJECT_ROOT)
    indexer.run()


@app.command()
def serve(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to pano.yaml config file."),
) -> None:
    """Start the MCP server for codebase RAG queries with live re-indexing."""
    from indexer import Indexer
    from mcp_server import create_server
    from watcher import Watcher

    cfg = load_config(config)

    indexer = Indexer(cfg, PROJECT_ROOT)
    file_watcher = Watcher(indexer, cfg, PROJECT_ROOT)
    watcher_thread = threading.Thread(target=file_watcher.run, daemon=True)
    watcher_thread.start()

    transport = cfg.get("mcp", {}).get("transport", "stdio")
    server = create_server(cfg, PROJECT_ROOT)
    server.run(transport=transport)


@app.command()
def watch(
    config: Optional[Path] = typer.Option(None, "--config", "-c", help="Path to pano.yaml config file."),
) -> None:
    """Watch the codebase for changes and re-index automatically."""
    from indexer import Indexer
    from watcher import Watcher

    cfg = load_config(config)
    indexer = Indexer(cfg, PROJECT_ROOT)
    file_watcher = Watcher(indexer, cfg, PROJECT_ROOT)
    file_watcher.run()


if __name__ == "__main__":
    app()
