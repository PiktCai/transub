from __future__ import annotations

import os
import stat
import tempfile
import unittest
from pathlib import Path

from transub.auth import AuthManager, resolve_llm_credentials
from transub.config import LLMConfig, PipelineConfig
from transub.translate import LLMTranslator


class AuthManagerTest(unittest.TestCase):
    def test_save_and_load_provider_credentials(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auth.toml"
            manager = AuthManager(path)

            manager.save_provider(
                "openai",
                api_key=" sk-test ",
                api_base=" https://api.example.com/v1/ ",
            )
            credentials = manager.load_provider("openai")

            self.assertEqual(credentials.api_key, "sk-test")
            self.assertEqual(credentials.api_base, "https://api.example.com/v1")
            mode = stat.S_IMODE(path.stat().st_mode)
            self.assertEqual(mode, 0o600)

    def test_environment_key_takes_precedence_over_auth_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "auth.toml"
            manager = AuthManager(path)
            manager.save_provider("openai", api_key="file-key")
            os.environ["TRANSUB_TEST_KEY"] = "env-key"
            self.addCleanup(lambda: os.environ.pop("TRANSUB_TEST_KEY", None))

            credentials = resolve_llm_credentials(
                provider="openai",
                api_key_env="TRANSUB_TEST_KEY",
                auth_manager=manager,
            )

            self.assertEqual(credentials.api_key, "env-key")


class LLMTranslatorAuthFallbackTest(unittest.TestCase):
    def test_translator_uses_auth_file_when_env_key_is_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            auth_path = Path(tmp) / "auth.toml"
            manager = AuthManager(auth_path)
            manager.save_provider(
                "openai",
                api_key="file-key",
                api_base="https://api.example.com/v1",
            )
            os.environ["TRANSUB_AUTH"] = str(auth_path)
            os.environ.pop("MISSING_TRANSLATOR_KEY", None)
            self.addCleanup(lambda: os.environ.pop("TRANSUB_AUTH", None))

            translator = LLMTranslator(
                config=LLMConfig(
                    provider="openai",
                    api_key_env="MISSING_TRANSLATOR_KEY",
                ),
                pipeline=PipelineConfig(),
            )

            self.assertEqual(translator.api_key, "file-key")
            self.assertEqual(
                translator.endpoint,
                "https://api.example.com/v1/chat/completions",
            )


if __name__ == "__main__":
    unittest.main()
