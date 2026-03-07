"""MCP server exposing codebase RAG search over a local LanceDB index."""

from pathlib import Path

import lancedb
from mcp.server.fastmcp import FastMCP


def create_server(config: dict, project_root: Path) -> FastMCP:
    """Build a configured FastMCP server with codebase search tools."""
    db_cfg = config.get("database", {})
    db_path = str(project_root / db_cfg.get("path", "codebase-db"))
    table_name = db_cfg.get("table_name", "code_chunks")

    mcp_cfg = config.get("mcp", {})
    server_name = mcp_cfg.get("server_name", "pano-codebase-rag")

    mcp = FastMCP(server_name)

    def _open_table():
        db = lancedb.connect(db_path)
        return db.open_table(table_name)

    @mcp.tool()
    def search_codebase(query: str, limit: int = 5) -> str:
        """Search the indexed codebase by semantic similarity.

        Args:
            query: Natural language or code search query.
            limit: Maximum number of chunks to return.
        """
        table = _open_table()
        results = table.search(query).limit(limit).to_list()

        if not results:
            return "No results found."

        parts: list[str] = []
        for r in results:
            header = (
                f"## {r['filename']}  "
                f"(lines {r['start_line']}-{r['end_line']}, {r['language']})"
            )
            parts.append(f"{header}\n```{r['language']}\n{r['text']}```")
        return "\n\n".join(parts)

    @mcp.tool()
    def get_file_context(filename: str) -> str:
        """Retrieve all indexed chunks for a specific file, ordered by line number.

        Args:
            filename: File path relative to the project root (e.g. "src/main.py").
        """
        table = _open_table()
        df = table.to_pandas()
        file_chunks = df[df["filename"] == filename].sort_values("start_line")

        if file_chunks.empty:
            return f"No indexed chunks found for '{filename}'."

        parts: list[str] = []
        for _, r in file_chunks.iterrows():
            header = f"## lines {r['start_line']}-{r['end_line']}"
            parts.append(f"{header}\n```{r['language']}\n{r['text']}```")
        return f"# {filename}\n\n" + "\n\n".join(parts)

    @mcp.tool()
    def list_indexed_files() -> str:
        """List all files that have been indexed in the codebase database."""
        table = _open_table()
        df = table.to_pandas()
        files = sorted(df["filename"].unique())

        if len(files) == 0:
            return "No files indexed."
        return "\n".join(files)

    return mcp
