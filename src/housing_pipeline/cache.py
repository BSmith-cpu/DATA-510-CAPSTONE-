"""Download-and-cache layer.

Upstream files here range from a 500KB Zillow CSV to a 300MB QCEW archive.
Re-downloading those on every run is what made the old pipeline painful to
operate, so every fetch goes through this module and lands in a gitignored
cache directory. A second run costs nothing.
"""

from __future__ import annotations

import logging
import shutil
import urllib.request
from pathlib import Path

from .config import CACHE_DIR, USER_AGENT

log = logging.getLogger(__name__)


class FetchError(RuntimeError):
    """Raised when a remote file could not be retrieved and no fallback exists."""


def cache_path(filename: str) -> Path:
    return CACHE_DIR / filename


def fetch(
    url: str,
    filename: str,
    *,
    refresh: bool = False,
    fallback: Path | None = None,
    timeout: int = 600,
) -> Path:
    """Return a local path to `url`, downloading it only when necessary.

    Parameters
    ----------
    url:
        Remote location to download from.
    filename:
        Name to store it under inside the cache directory.
    refresh:
        Re-download even if a cached copy exists.
    fallback:
        A committed local file to use if the download fails. Several upstream
        hosts (FHFA especially) change their download paths periodically; a
        fallback keeps the pipeline runnable when that happens.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    dest = cache_path(filename)

    if dest.exists() and not refresh:
        log.debug("cache hit: %s", filename)
        return dest

    log.info("fetching %s", url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest.with_suffix(dest.suffix + ".part")
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            if response.status != 200:
                raise FetchError(f"{url} returned HTTP {response.status}")
            with open(tmp, "wb") as handle:
                shutil.copyfileobj(response, handle)
        tmp.replace(dest)
        return dest
    except Exception as exc:  # noqa: BLE001 - we deliberately fall back on any failure
        tmp.unlink(missing_ok=True)
        if fallback is not None and Path(fallback).exists():
            log.warning(
                "fetch failed for %s (%s); using committed fallback %s",
                url, exc, fallback,
            )
            return Path(fallback)
        raise FetchError(f"could not fetch {url}: {exc}") from exc


def clear(filename: str | None = None) -> None:
    """Delete one cached file, or the whole cache when `filename` is None."""
    if filename is None:
        if CACHE_DIR.exists():
            shutil.rmtree(CACHE_DIR)
        return
    cache_path(filename).unlink(missing_ok=True)
