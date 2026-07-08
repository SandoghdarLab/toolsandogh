import multiprocessing
import os
import sys
from pathlib import Path

import jax
import pytest

# Enable JAX's persistent compilation cache so that repeated test runs skip JIT
# recompilation. The first run populates the cache at no extra cost; subsequent
# runs reuse it (empirically ~2.4x faster for the simulation-heavy tests).
#
# The cache directory honors, in order:
#   1. an explicit ``JAX_COMPILATION_CACHE_DIR`` environment variable (used by
#      CI for a deterministic, uniformly cacheable path), else
#   2. the platform-canonical cache root (see ``_default_cache_root``).
#
# ``min_compile_time_secs=0`` persists *every* compilation, not just those
# longer than the 1s default; the footprint is small (~a few MB) and there is
# no cold-run penalty. The cache key includes the jaxlib version, platform, and
# XLA flags, so entries never collide across Python versions, OSes, or
# dependency resolutions.


def _default_cache_root() -> str:
    """Return the platform-canonical user cache directory."""
    home = str(Path.home())
    if sys.platform == "win32":
        return os.environ.get("LOCALAPPDATA") or os.path.join(home, "AppData", "Local")
    if sys.platform == "darwin":
        return os.path.join(home, "Library", "Caches")
    return os.environ.get("XDG_CACHE_HOME") or os.path.join(home, ".cache")


_env_cache = os.environ.get("JAX_COMPILATION_CACHE_DIR")
_cache_dir = Path(
    os.path.expanduser(_env_cache)
    if _env_cache
    else os.path.join(_default_cache_root(), "jax-compilation-cache")
)
_cache_dir.mkdir(parents=True, exist_ok=True)
jax.config.update("jax_compilation_cache_dir", str(_cache_dir))
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)


@pytest.fixture(scope="session", autouse=True)
def always_spawn():
    multiprocessing.set_start_method("spawn")
