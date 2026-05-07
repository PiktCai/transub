from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

try:  # Python 3.11+
    import tomllib as tomli  # type: ignore[import-not-found]
except ModuleNotFoundError:  # pragma: no cover - fallback for <3.11
    import tomli  # type: ignore[no-redef]

import tomli_w


AUTH_FILENAME = "auth.toml"


@dataclass
class AuthCredentials:
    api_key: str | None = None
    api_base: str | None = None

    def is_configured(self) -> bool:
        return bool(self.api_key)


class AuthManager:
    """Read and write local provider credentials."""

    def __init__(self, path: Path | None = None):
        self.path = path or self.default_path()

    @classmethod
    def default_path(cls) -> Path:
        env_path = os.getenv("TRANSUB_AUTH")
        if env_path:
            return Path(env_path).expanduser()
        return Path.home() / ".transub" / AUTH_FILENAME

    def load_all(self) -> dict:
        if not self.path.exists():
            return {}
        with self.path.open("rb") as fh:
            raw = tomli.load(fh)
        return raw if isinstance(raw, dict) else {}

    def load_provider(self, provider: str) -> AuthCredentials:
        raw = self.load_all().get(provider, {})
        if not isinstance(raw, dict):
            return AuthCredentials()
        return AuthCredentials(
            api_key=_clean_string(raw.get("api_key")),
            api_base=_clean_string(raw.get("api_base")),
        )

    def save_provider(
        self,
        provider: str,
        *,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> None:
        data = self.load_all()
        provider_data = data.get(provider, {})
        if not isinstance(provider_data, dict):
            provider_data = {}

        if api_key is not None:
            cleaned_key = api_key.strip()
            if cleaned_key:
                provider_data["api_key"] = cleaned_key
            else:
                provider_data.pop("api_key", None)

        if api_base is not None:
            cleaned_base = api_base.strip()
            if cleaned_base:
                provider_data["api_base"] = cleaned_base.rstrip("/")
            else:
                provider_data.pop("api_base", None)

        if provider_data:
            data[provider] = provider_data
        else:
            data.pop(provider, None)

        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("wb") as fh:
            tomli_w.dump(data, fh)
        try:
            os.chmod(self.path, 0o600)
        except OSError:
            pass


def resolve_llm_credentials(
    *,
    provider: str,
    api_key_env: str,
    auth_manager: AuthManager | None = None,
) -> AuthCredentials:
    env_key = _clean_string(os.getenv(api_key_env))
    if env_key:
        return AuthCredentials(api_key=env_key)
    manager = auth_manager or AuthManager()
    return manager.load_provider(provider)


def _clean_string(value: object) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


__all__ = [
    "AUTH_FILENAME",
    "AuthCredentials",
    "AuthManager",
    "resolve_llm_credentials",
]
