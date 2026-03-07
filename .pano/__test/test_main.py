import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import yaml
from typer.testing import CliRunner

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from main import app, load_config

runner = CliRunner()


class TestLoadConfig:
    """Config loading from YAML files."""

    def test_loads_valid_config(self, config_yaml_file: Path):
        cfg = load_config(config_yaml_file)

        assert isinstance(cfg, dict)
        assert "indexer" in cfg
        assert "database" in cfg

    def test_missing_file_exits(self, tmp_path: Path):
        from click.exceptions import Exit

        with pytest.raises(Exit):
            load_config(tmp_path / "nonexistent.yaml")

    def test_empty_file_exits(self, tmp_path: Path):
        from click.exceptions import Exit

        empty = tmp_path / "empty.yaml"
        empty.write_text("", encoding="utf-8")

        with pytest.raises(Exit):
            load_config(empty)

    def test_malformed_file_exits(self, tmp_path: Path):
        from click.exceptions import Exit

        bad = tmp_path / "bad.yaml"
        bad.write_text("just a string", encoding="utf-8")

        with pytest.raises(Exit):
            load_config(bad)


class TestIndexCommand:
    """CLI 'index' command invokes the Indexer."""

    def test_index_runs(self, config_yaml_file: Path):
        mock_instance = MagicMock()

        with patch("indexer.Indexer", return_value=mock_instance) as MockIndexer:
            result = runner.invoke(app, ["index", "--config", str(config_yaml_file)])

        assert result.exit_code == 0
        MockIndexer.assert_called_once()
        mock_instance.run.assert_called_once()

    def test_index_with_custom_config(self, config_yaml_file: Path):
        mock_instance = MagicMock()

        with patch("indexer.Indexer", return_value=mock_instance):
            result = runner.invoke(app, ["index", "--config", str(config_yaml_file)])

        assert result.exit_code == 0

    def test_index_missing_config_fails(self, tmp_path: Path):
        result = runner.invoke(app, ["index", "--config", str(tmp_path / "nope.yaml")])

        assert result.exit_code != 0


class TestServeCommand:
    """CLI 'serve' command starts the MCP server."""

    def test_serve_runs(self, config_yaml_file: Path):
        mock_server = MagicMock()

        with patch("mcp_server.create_server", return_value=mock_server) as mock_create:
            result = runner.invoke(app, ["serve", "--config", str(config_yaml_file)])

        assert result.exit_code == 0
        mock_create.assert_called_once()
        mock_server.run.assert_called_once_with(transport="stdio")

    def test_serve_missing_config_fails(self, tmp_path: Path):
        result = runner.invoke(app, ["serve", "--config", str(tmp_path / "nope.yaml")])

        assert result.exit_code != 0


class TestHelpOutput:
    """CLI help lists all available commands."""

    def test_help_shows_commands(self):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "index" in result.output
        assert "serve" in result.output

    def test_index_help(self):
        result = runner.invoke(app, ["index", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
