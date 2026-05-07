from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Optional


_cache_dir: Optional[Path] = None
_enabled = True


def init_cache(cache_dir: Optional[str] = None) -> Path:
    """Initialize the cache directory."""
    global _cache_dir
    if cache_dir:
        _cache_dir = Path(cache_dir)
    else:
        _cache_dir = Path.home() / ".cache" / "transub" / "api_cache"
    _cache_dir.mkdir(parents=True, exist_ok=True)
    return _cache_dir


def get_cache_dir() -> Path:
    """Get the current cache directory."""
    if _cache_dir is None:
        return init_cache()
    return _cache_dir


def enable_cache() -> None:
    """Enable caching."""
    global _enabled
    _enabled = True


def disable_cache() -> None:
    """Disable caching."""
    global _enabled
    _enabled = False


def is_cache_enabled() -> bool:
    """Check if caching is enabled."""
    return _enabled


def generate_cache_key(data: dict) -> str:
    """Generate a deterministic cache key from data."""
    serialized = json.dumps(data, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(serialized.encode()).hexdigest()


def get_cached(key: str) -> Optional[dict]:
    """Retrieve cached response by key."""
    if not _enabled:
        return None

    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{key}.json"

    if not cache_file.exists():
        return None

    try:
        with cache_file.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def set_cached(key: str, data: dict) -> None:
    """Store response in cache."""
    if not _enabled:
        return

    cache_dir = get_cache_dir()
    cache_file = cache_dir / f"{key}.json"

    try:
        with cache_file.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except OSError:
        pass


def clear_cache() -> int:
    """Clear all cached responses. Returns number of files removed."""
    cache_dir = get_cache_dir()
    count = 0
    for cache_file in cache_dir.glob("*.json"):
        try:
            cache_file.unlink()
            count += 1
        except OSError:
            pass
    return count


def get_cache_stats() -> dict:
    """Get cache statistics."""
    cache_dir = get_cache_dir()
    files = list(cache_dir.glob("*.json"))
    total_size = sum(f.stat().st_size for f in files if f.exists())
    return {
        "count": len(files),
        "size_bytes": total_size,
        "size_mb": round(total_size / (1024 * 1024), 2),
        "directory": str(cache_dir),
    }


__all__ = [
    "init_cache",
    "get_cache_dir",
    "enable_cache",
    "disable_cache",
    "is_cache_enabled",
    "generate_cache_key",
    "get_cached",
    "set_cached",
    "clear_cache",
    "get_cache_stats",
]
