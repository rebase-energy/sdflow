"""Environment variables and cache directories, with the pre-rename spellings honoured.

sdflow was emflow until 0.1.0. Shell profiles, CI jobs and sandbox images still
export ``EMFLOW_*`` variables, and warmed caches still live under
``~/.cache/emflow``. Every lookup here tries the ``SDFLOW_`` name first and the
``EMFLOW_`` name second, and a cache directory that only exists under the old
root is used as is, so nothing already configured or downloaded is orphaned by
the rename.
"""

from __future__ import annotations

import os
from pathlib import Path
import typing as t

PREFIX = "SDFLOW_"
LEGACY_PREFIX = "EMFLOW_"
CACHE_HOME = Path.home() / ".cache"


def getenv(name: str, default: t.Optional[str] = None) -> t.Optional[str]:
    """``$SDFLOW_<name>``, else ``$EMFLOW_<name>``, else ``default``.

    ``name`` is the bare suffix (``"GRID_API_KEY"``). An empty value counts as
    unset, like a missing variable.
    """
    for prefix in (PREFIX, LEGACY_PREFIX):
        value = os.environ.get(prefix + name)
        if value:
            return value
    return default


def cache_root(sub: str, env_name: t.Optional[str] = None) -> Path:
    """Cache directory for the ``sub`` component (``"grid"``, ``"gefcom2014"``).

    Resolution order: ``$SDFLOW_<ENV_NAME>`` / ``$EMFLOW_<ENV_NAME>`` (default
    ``<SUB>_CACHE``), then ``~/.cache/sdflow/<sub>``, except that when only
    ``~/.cache/emflow/<sub>`` exists on disk that directory is returned, so a
    cache warmed before the rename keeps being used.
    """
    env_name = env_name or f"{sub.upper()}_CACHE"
    configured = getenv(env_name)
    if configured:
        return Path(configured).expanduser()
    new, old = CACHE_HOME / "sdflow" / sub, CACHE_HOME / "emflow" / sub
    if not new.exists() and old.exists():
        return old
    return new
