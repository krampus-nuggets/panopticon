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
    """CLI 'serve' command starts the MCP server with background watcher."""

    @patch("watcher.Watcher")
    @patch("mcp_server.create_server")
    def test_serve_runs_with_watcher(
        self, mock_create, mock_watcher_cls, config_yaml_file: Path
    ):
        mock_server = MagicMock()
        mock_create.return_value = mock_server
        mock_watcher_instance = MagicMock()
        mock_watcher_cls.return_value = mock_watcher_instance

        with patch("main.threading.Thread") as mock_thread:
            mock_thread_instance = MagicMock()
            mock_thread.return_value = mock_thread_instance

            result = runner.invoke(app, ["serve", "--config", str(config_yaml_file)])

        assert result.exit_code == 0
        mock_thread.assert_called_once_with(
            target=mock_watcher_instance.run, daemon=True
        )
        mock_thread_instance.start.assert_called_once()
        mock_create.assert_called_once()
        mock_server.run.assert_called_once_with(transport="stdio")

    def test_serve_missing_config_fails(self, tmp_path: Path):
        result = runner.invoke(app, ["serve", "--config", str(tmp_path / "nope.yaml")])

        assert result.exit_code != 0


class TestWatchCommand:
    """CLI 'watch' command starts the standalone file watcher."""

    @patch("watcher.Watcher")
    def test_watch_runs(self, mock_watcher_cls, config_yaml_file: Path):
        mock_watcher_instance = MagicMock()
        mock_watcher_cls.return_value = mock_watcher_instance

        result = runner.invoke(app, ["watch", "--config", str(config_yaml_file)])

        assert result.exit_code == 0
        mock_watcher_cls.assert_called_once()
        mock_watcher_instance.run.assert_called_once()

    def test_watch_missing_config_fails(self, tmp_path: Path):
        result = runner.invoke(app, ["watch", "--config", str(tmp_path / "nope.yaml")])

        assert result.exit_code != 0


class TestHelpOutput:
    """CLI help lists all available commands."""

    def test_help_shows_commands(self):
        result = runner.invoke(app, ["--help"])

        assert result.exit_code == 0
        assert "index" in result.output
        assert "serve" in result.output
        assert "watch" in result.output

    def test_index_help(self):
        result = runner.invoke(app, ["index", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output

    def test_serve_help(self):
        result = runner.invoke(app, ["serve", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output

    def test_watch_help(self):
        result = runner.invoke(app, ["watch", "--help"])

        assert result.exit_code == 0
        assert "--config" in result.output
