from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from conftest import MOCK_VECTOR
from mcp_server import create_server


@pytest.fixture()
def mock_table():
    """A mock LanceDB table with sample chunk data."""
    data = pd.DataFrame([
        {
            "filename": "src/app.py",
            "start_line": 1,
            "end_line": 5,
            "language": "python",
            "text": "def hello():\n    return 'world'\n",
        },
        {
            "filename": "src/utils.py",
            "start_line": 1,
            "end_line": 10,
            "language": "python",
            "text": "import os\n",
        },
    ])

    table = MagicMock()
    table.to_pandas.return_value = data

    search_result = MagicMock()
    search_result.limit.return_value = search_result
    search_result.to_list.return_value = data.to_dict("records")
    table.search.return_value = search_result

    return table


def _get_tool_fn(server, name: str):
    """Extract a tool function by name from a FastMCP server."""
    for tool in server._tool_manager.list_tools():
        if tool.name == name:
            return tool.fn
    return None


class TestCreateServer:
    """Factory builds a properly configured FastMCP instance."""

    @patch("mcp_server.lancedb")
    def test_returns_fastmcp(self, mock_lancedb, sample_config: dict, tmp_path: Path):
        server = create_server(sample_config, tmp_path)

        from mcp.server.fastmcp import FastMCP
        assert isinstance(server, FastMCP)

    @patch("mcp_server.lancedb")
    def test_server_name_from_config(self, mock_lancedb, sample_config: dict, tmp_path: Path):
        sample_config["mcp"]["server_name"] = "custom-name"
        server = create_server(sample_config, tmp_path)

        assert server.name == "custom-name"


class TestSearchCodebase:
    """search_codebase tool returns formatted results."""

    @patch("mcp_server.ollama")
    @patch("mcp_server.lancedb")
    def test_returns_formatted_results(
        self, mock_lancedb, mock_ollama, sample_config: dict, tmp_path: Path, mock_table
    ):
        mock_lancedb.connect.return_value.open_table.return_value = mock_table
        mock_ollama.embed.return_value = {"embeddings": [MOCK_VECTOR]}

        server = create_server(sample_config, tmp_path)
        tool_fn = _get_tool_fn(server, "search_codebase")

        assert tool_fn is not None
        result = tool_fn("hello", limit=5)

        assert "src/app.py" in result
        assert "python" in result
        mock_ollama.embed.assert_called_once()

    @patch("mcp_server.ollama")
    @patch("mcp_server.lancedb")
    def test_empty_results(self, mock_lancedb, mock_ollama, sample_config: dict, tmp_path: Path):
        empty_table = MagicMock()
        search_result = MagicMock()
        search_result.limit.return_value = search_result
        search_result.to_list.return_value = []
        empty_table.search.return_value = search_result
        mock_lancedb.connect.return_value.open_table.return_value = empty_table
        mock_ollama.embed.return_value = {"embeddings": [MOCK_VECTOR]}

        server = create_server(sample_config, tmp_path)
        tool_fn = _get_tool_fn(server, "search_codebase")

        result = tool_fn("nothing", limit=5)
        assert result == "No results found."


class TestGetFileContext:
    """get_file_context tool returns file-specific chunks."""

    @patch("mcp_server.lancedb")
    def test_returns_file_chunks(
        self, mock_lancedb, sample_config: dict, tmp_path: Path, mock_table
    ):
        mock_lancedb.connect.return_value.open_table.return_value = mock_table

        server = create_server(sample_config, tmp_path)
        tool_fn = _get_tool_fn(server, "get_file_context")

        assert tool_fn is not None
        result = tool_fn("src/app.py")

        assert "src/app.py" in result
        assert "lines 1-5" in result

    @patch("mcp_server.lancedb")
    def test_missing_file(self, mock_lancedb, sample_config: dict, tmp_path: Path):
        empty_df = pd.DataFrame(columns=["filename", "start_line", "end_line", "language", "text"])
        empty_table = MagicMock()
        empty_table.to_pandas.return_value = empty_df
        mock_lancedb.connect.return_value.open_table.return_value = empty_table

        server = create_server(sample_config, tmp_path)
        tool_fn = _get_tool_fn(server, "get_file_context")

        result = tool_fn("nonexistent.py")
        assert "No indexed chunks found" in result


class TestListIndexedFiles:
    """list_indexed_files tool returns sorted filenames."""

    @patch("mcp_server.lancedb")
    def test_lists_files(
        self, mock_lancedb, sample_config: dict, tmp_path: Path, mock_table
    ):
        mock_lancedb.connect.return_value.open_table.return_value = mock_table

        server = create_server(sample_config, tmp_path)
        tool_fn = _get_tool_fn(server, "list_indexed_files")

        assert tool_fn is not None
        result = tool_fn()

        assert "src/app.py" in result
        assert "src/utils.py" in result

    @patch("mcp_server.lancedb")
    def test_no_files_indexed(self, mock_lancedb, sample_config: dict, tmp_path: Path):
        empty_df = pd.DataFrame(columns=["filename"])
        empty_table = MagicMock()
        empty_table.to_pandas.return_value = empty_df
        mock_lancedb.connect.return_value.open_table.return_value = empty_table

        server = create_server(sample_config, tmp_path)
        tool_fn = _get_tool_fn(server, "list_indexed_files")

        result = tool_fn()
        assert result == "No files indexed."
