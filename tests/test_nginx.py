"""Tests for nginx config generation."""

import pytest
from pathlib import Path

from vllmd.nginx import NginxManager


def make_manager(tmp_path: Path, lb_port: int = 9000) -> NginxManager:
    # Use a fake binary path — we only test config generation here
    return NginxManager(tmp_path, lb_port, nginx_bin="/usr/bin/nginx")


class TestNginxConfigGeneration:
    def test_empty_backends_has_down_server(self, tmp_path):
        mgr = make_manager(tmp_path)
        config = mgr._render_config([])
        assert "server 127.0.0.1:1 down;" in config
        assert "least_conn" not in config

    def test_single_backend(self, tmp_path):
        mgr = make_manager(tmp_path)
        config = mgr._render_config([("nid001", 8000)])
        assert "server nid001:8000;" in config
        assert "least_conn;" in config

    def test_multiple_backends(self, tmp_path):
        mgr = make_manager(tmp_path)
        backends = [("nid001", 8000), ("nid002", 8001), ("nid003", 8002)]
        config = mgr._render_config(backends)
        for host, port in backends:
            assert f"server {host}:{port};" in config
        assert config.count("server nid") == 3

    def test_listen_port(self, tmp_path):
        mgr = make_manager(tmp_path, lb_port=9876)
        config = mgr._render_config([("nid001", 8000)])
        assert "listen 9876;" in config

    def test_streaming_options(self, tmp_path):
        mgr = make_manager(tmp_path)
        config = mgr._render_config([("nid001", 8000)])
        assert "proxy_buffering    off;" in config
        assert "proxy_cache        off;" in config

    def test_long_timeouts(self, tmp_path):
        mgr = make_manager(tmp_path)
        config = mgr._render_config([("nid001", 8000)])
        assert "proxy_send_timeout    600s;" in config
        assert "proxy_read_timeout    600s;" in config

    def test_config_written_to_file(self, tmp_path):
        mgr = make_manager(tmp_path)
        mgr._write_config([("nid001", 8000)])
        assert mgr.conf_path.exists()
        content = mgr.conf_path.read_text()
        assert "server nid001:8000;" in content

    def test_pid_file_in_session_dir(self, tmp_path):
        mgr = make_manager(tmp_path)
        config = mgr._render_config([])
        assert str(tmp_path) in config  # pid file path contains session dir

    def test_temp_dirs_in_session_dir(self, tmp_path):
        mgr = make_manager(tmp_path)
        config = mgr._render_config([])
        assert "client_body_temp_path" in config
        assert str(tmp_path) in config
