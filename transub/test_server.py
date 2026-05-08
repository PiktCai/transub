from __future__ import annotations

import importlib.util
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch


@unittest.skipUnless(importlib.util.find_spec("fastapi"), "server extra is not installed")
class TestDesktopServer(unittest.TestCase):
    def setUp(self) -> None:
        from transub import server

        self.server = server
        self.client = server.app
        server._active_pipeline = None
        server._pipeline_cancelled = False
        server._pipeline_status.clear()
        server._pipeline_status.update({"state": "idle"})

    def tearDown(self) -> None:
        self.server._active_pipeline = None
        self.server._pipeline_cancelled = False
        self.server._pipeline_status.clear()
        self.server._pipeline_status.update({"state": "idle"})

    def test_run_rejects_missing_video(self) -> None:
        from fastapi.testclient import TestClient

        with TestClient(self.client) as client:
            response = client.post("/api/run", json={"video_path": "/definitely/not/here.mp4"})

        self.assertEqual(response.status_code, 400)
        self.assertIn("Video not found", response.json()["detail"])

    def test_run_returns_debug_log_path(self) -> None:
        from fastapi.testclient import TestClient

        started = threading.Event()

        def fake_run(*_args, **_kwargs):
            started.set()

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample video.mp4"
            video.write_bytes(b"video")
            with patch.object(self.server, "_run_pipeline_thread", side_effect=fake_run):
                with TestClient(self.client) as client:
                    response = client.post(
                        "/api/run",
                        json={"video_path": str(video), "work_dir": tmp, "transcribe_only": True},
                    )

            payload = response.json()

        self.assertEqual(response.status_code, 200)
        self.assertEqual(payload["status"], "started")
        self.assertIn("log_path", payload)
        self.assertTrue(payload["log_path"].endswith(".log"))

    def test_run_rejects_second_active_pipeline(self) -> None:
        from fastapi.testclient import TestClient

        with tempfile.TemporaryDirectory() as tmp:
            video = Path(tmp) / "sample.mp4"
            video.write_bytes(b"video")
            sleeper = threading.Event()
            active_thread = threading.Thread(target=sleeper.wait)
            active_thread.start()
            self.server._active_pipeline = active_thread
            try:
                with TestClient(self.client) as client:
                    response = client.post("/api/run", json={"video_path": str(video), "work_dir": tmp})
            finally:
                sleeper.set()
                active_thread.join(timeout=1)

        self.assertEqual(response.status_code, 409)
        self.assertIn("already running", response.json()["detail"])


if __name__ == "__main__":
    unittest.main()
