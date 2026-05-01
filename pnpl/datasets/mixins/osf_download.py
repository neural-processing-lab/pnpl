"""
OSFDownloadMixin - Provides Open Science Framework (OSF) download functionality.

OSF projects (https://osf.io) host research datasets via the OSF Storage
provider. Public projects need no authentication. This mixin downloads
individual files on demand, mirroring the surface of ``OhanaDownloadMixin``:

  - ``ensure_file(local_path)`` — download if missing, return local path.
  - ``prefetch_files(local_paths)`` — parallel download of a batch.
  - ``get_dataset_manifest(refresh=False)`` — return raw manifest payload.
  - ``list_remote_files(refresh=False)`` — return dataset-relative paths.

Multi-component projects (e.g. MEG-MASC, where 27 subjects span four sibling
OSF nodes because of OSF's per-component storage cap) are supported via
``OSF_PROJECT_FALLBACKS``: the manifest aggregates files from every
configured node, with the primary node winning on duplicate paths.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Optional
from urllib.parse import quote

import requests
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout


class OSFDownloadMixin:
    """
    Mixin providing OSF download functionality.

    Classes using this mixin should define:
    - ``OSF_PROJECT_ID``: str — primary OSF node id (e.g. ``"ag3kj"``)

    Optional class attributes:
    - ``OSF_PROJECT_FALLBACKS``: list[str] — additional OSF node ids whose
      ``osfstorage`` is searched after the primary. Useful when a single
      logical dataset spans multiple OSF components.
    - ``OSF_API_BASE``: str — defaults to ``https://api.osf.io/v2``.
    - ``OSF_FILES_BASE``: str — defaults to ``https://files.osf.io/v1``.
    - ``OSF_TOKEN_ENV``: str — env var name for an optional OAuth token
      (default ``"OSF_TOKEN"``). Public projects do not need auth.

    Expected instance attributes:
    - ``data_path``: str — local data directory (manifest paths are
      resolved relative to this).
    - ``download``: bool — whether downloading is enabled.
    """

    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)
    _download_futures: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _manifest_cache: ClassVar[dict] = {}
    _folder_listing_cache: ClassVar[dict] = {}
    _print_lock: ClassVar[threading.Lock] = threading.Lock()

    OSF_PROJECT_ID: ClassVar[str] = ""
    OSF_PROJECT_FALLBACKS: ClassVar[list[str]] = []
    OSF_API_BASE: ClassVar[str] = "https://api.osf.io/v2"
    OSF_FILES_BASE: ClassVar[str] = "https://files.osf.io/v1"
    OSF_TOKEN_ENV: ClassVar[str] = "OSF_TOKEN"

    # ------------------------------------------------------------------
    # Helpers shared with OhanaDownloadMixin's progress UI
    # ------------------------------------------------------------------

    @staticmethod
    def _is_truthy(v: Optional[str]) -> bool:
        if v is None:
            return False
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def _progress_enabled(cls) -> bool:
        v = os.getenv("PNPL_OSF_PROGRESS")
        if v is None:
            return True
        return cls._is_truthy(v)

    @staticmethod
    def _format_bytes(n: Optional[int]) -> str:
        if n is None:
            return "?"
        units = ["B", "KB", "MB", "GB", "TB"]
        f = float(n)
        u = 0
        while f >= 1024.0 and u < len(units) - 1:
            f /= 1024.0
            u += 1
        if u == 0:
            return f"{int(f)} {units[u]}"
        return f"{f:.2f} {units[u]}"

    @classmethod
    def _project_nodes(cls) -> list[str]:
        nodes = []
        if cls.OSF_PROJECT_ID:
            nodes.append(cls.OSF_PROJECT_ID)
        nodes.extend(cls.OSF_PROJECT_FALLBACKS or [])
        return nodes

    @classmethod
    def _osf_token(cls) -> Optional[str]:
        return os.getenv(cls.OSF_TOKEN_ENV) or None

    @classmethod
    def _request_headers(cls) -> dict:
        headers = {"Accept": "application/vnd.api+json"}
        token = cls._osf_token()
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    # ------------------------------------------------------------------
    # Public surface used by datasets
    # ------------------------------------------------------------------

    def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files in parallel (skips already-present)."""
        futures = []
        for fpath in {p for p in file_paths if not os.path.exists(p)}:
            futures.append(self._schedule_download(fpath))

        if futures:
            print(f"Downloading {len(futures)} files from OSF...")
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading a file: {e}")
            print("Done!")

    def ensure_file(self, fpath: str) -> str:
        """Ensure a file exists locally, downloading from OSF if needed."""
        if os.path.exists(fpath):
            return fpath

        if not getattr(self, "download", True):
            raise FileNotFoundError(f"File not found: {fpath}. Download is disabled.")

        future = self._schedule_download(fpath)
        return future.result()

    def _schedule_download(self, fpath: str):
        rel_path = os.path.relpath(fpath, self.data_path).replace(os.path.sep, "/")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with self._lock:
            if fpath not in self._download_futures:
                # Lazy lookup: walk only the folders on the path to this file
                # rather than building a global manifest of every file in
                # every configured node.
                entry = type(self).resolve_remote_file(rel_path)
                self._download_futures[fpath] = self._executor.submit(
                    self._download_with_retry_static,
                    fpath=fpath,
                    rel_path=rel_path,
                    entry=entry,
                    files_base=self.OSF_FILES_BASE.rstrip("/"),
                    token=self._osf_token(),
                )
            return self._download_futures[fpath]

    @classmethod
    def get_dataset_manifest(cls, refresh: bool = False) -> dict:
        """
        Build a manifest of every file on OSF storage for the configured
        node(s). Returns a dict keyed by relative path:

            {
                "sub-01/ses-0/meg/...": {
                    "node_id": "ag3kj",
                    "file_id": "63e66...",
                    "size": 953619,
                },
                ...
            }

        Walks the entire folder tree across every configured node, which
        can take minutes for large datasets. When the caller knows which
        files it needs, prefer :meth:`resolve_remote_file` — it walks only
        the folders along the path to a single file.
        """
        nodes = cls._project_nodes()
        if not nodes:
            raise RuntimeError(
                "OSF_PROJECT_ID is not set on the dataset class; cannot fetch OSF manifest."
            )

        cache_key = (cls.OSF_API_BASE.rstrip("/"), tuple(nodes), cls._osf_token() or "")
        if not refresh and cache_key in cls._manifest_cache:
            return cls._manifest_cache[cache_key]

        api_base = cls.OSF_API_BASE.rstrip("/")
        token = cls._osf_token()
        manifest: dict[str, dict] = {}

        # Surface what we're doing — the walk is silent otherwise and can
        # take minutes for multi-component projects like MEG-MASC.
        cls._log(
            f"OSF: fetching manifest from {len(nodes)} node(s) "
            f"({', '.join(nodes)})… one-time, may take a while."
        )

        for node_id in nodes:
            t0 = time.monotonic()
            count_before = len(manifest)
            for entry in cls._walk_node(node_id, api_base, token):
                rel_path = entry["path"]
                if rel_path in manifest:
                    # Primary node (first in list) wins.
                    continue
                manifest[rel_path] = {
                    "node_id": entry["node_id"],
                    "file_id": entry["file_id"],
                    "size": entry["size"],
                }
            cls._log(
                f"OSF: {node_id} — {len(manifest) - count_before} new files "
                f"in {time.monotonic() - t0:.1f}s ({len(manifest)} total)"
            )

        cls._manifest_cache[cache_key] = manifest
        return manifest

    @classmethod
    def _log(cls, message: str) -> None:
        """Best-effort progress line, gated on PNPL_OSF_PROGRESS like the
        download progress bar. Goes to stderr so it interleaves with
        download bars cleanly."""
        if not cls._progress_enabled():
            return
        with cls._print_lock:
            sys.stderr.write(message + "\n")
            sys.stderr.flush()

    @classmethod
    def list_remote_files(cls, refresh: bool = False) -> list[str]:
        """Return dataset-relative file paths advertised by the OSF manifest."""
        manifest = cls.get_dataset_manifest(refresh=refresh)
        return sorted(manifest.keys())

    # ------------------------------------------------------------------
    # Lazy per-file resolution (no global manifest)
    # ------------------------------------------------------------------

    @classmethod
    def resolve_remote_file(cls, rel_path: str) -> dict:
        """Resolve a single file's OSF location by walking only the
        folders on its path (~3-4 API calls per *new* folder, served
        from cache thereafter).

        Returns ``{"node_id", "file_id", "size"}``.

        If a global manifest has already been fetched, that entry wins —
        callers are still free to call :meth:`get_dataset_manifest` for
        bulk operations and benefit from the cache.
        """
        # Manifest hit?
        nodes = cls._project_nodes()
        if not nodes:
            raise RuntimeError(
                "OSF_PROJECT_ID is not set on the dataset class; cannot resolve from OSF."
            )
        cache_key = (cls.OSF_API_BASE.rstrip("/"), tuple(nodes), cls._osf_token() or "")
        cached_manifest = cls._manifest_cache.get(cache_key)
        if cached_manifest is not None and rel_path in cached_manifest:
            return cached_manifest[rel_path]

        api_base = cls.OSF_API_BASE.rstrip("/")
        token = cls._osf_token()
        parts = [p for p in rel_path.split("/") if p]
        if not parts:
            raise FileNotFoundError(f"Empty OSF path: {rel_path!r}")

        last_exc: Optional[BaseException] = None
        for node_id in nodes:
            try:
                entry = cls._lookup_in_node(node_id, parts, api_base, token)
            except FileNotFoundError as exc:
                last_exc = exc
                continue
            if entry is not None:
                return entry

        raise FileNotFoundError(
            f"OSF file not found in any of {nodes}: '{rel_path}'."
        ) from last_exc

    @classmethod
    def _lookup_in_node(
        cls,
        node_id: str,
        parts: list[str],
        api_base: str,
        token: Optional[str],
    ) -> Optional[dict]:
        """Walk down ``node_id``'s osfstorage to find ``parts[0]/parts[1]/...``.
        Returns the file entry or None if any path component is missing.
        Folder listings are cached per (node, url) so subsequent lookups
        of siblings under the same parent are free."""
        folder_url = f"{api_base}/nodes/{quote(node_id, safe='')}/files/osfstorage/"

        for i, part in enumerate(parts):
            listing = cls._list_folder(node_id, folder_url, token)
            match = next((e for e in listing if e["name"] == part), None)
            if match is None:
                return None
            if i == len(parts) - 1:
                if match["kind"] != "file":
                    return None
                return {
                    "node_id": node_id,
                    "file_id": match["file_id"],
                    "size": match["size"],
                }
            if match["kind"] != "folder" or not match["files_url"]:
                return None
            folder_url = match["files_url"]
        return None

    @classmethod
    def _list_folder(
        cls,
        node_id: str,
        url: str,
        token: Optional[str],
    ) -> list[dict]:
        """List immediate children of an OSF folder URL. Cached per
        (node, url, token)."""
        cache = cls._folder_listing_cache
        cache_key = (node_id, url, token or "")
        cached = cache.get(cache_key)
        if cached is not None:
            return cached

        headers = {"Accept": "application/vnd.api+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        entries: list[dict] = []
        next_url: Optional[str] = (
            url + ("?" if "?" not in url else "&") + "page%5Bsize%5D=100"
        )
        while next_url:
            payload = cls._get_with_retry(next_url, headers)
            for entry in payload.get("data", []):
                attrs = entry.get("attributes", {}) or {}
                name = attrs.get("name")
                if not name:
                    continue
                files_url = (
                    ((entry.get("relationships") or {}).get("files") or {})
                    .get("links", {}).get("related", {}).get("href")
                )
                entries.append({
                    "kind": attrs.get("kind"),
                    "name": name,
                    "file_id": entry.get("id"),
                    "size": attrs.get("size"),
                    "files_url": files_url,
                })
            next_url = (payload.get("links") or {}).get("next")

        cache[cache_key] = entries
        return entries

    # ------------------------------------------------------------------
    # Manifest construction — recursive folder walk
    # ------------------------------------------------------------------

    @classmethod
    def _get_with_retry(
        cls,
        url: str,
        headers: dict,
        max_retries: int = 5,
        timeout_s: int = 30,
    ) -> dict:
        """GET an OSF API endpoint and return parsed JSON. Retries 429/5xx
        and transient network errors with exponential backoff."""
        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                resp = requests.get(url, headers=headers, timeout=timeout_s)
                if resp.status_code == 404:
                    raise FileNotFoundError(
                        f"OSF API 404 for {url}. Check OSF_PROJECT_ID / file id."
                    )
                if resp.status_code == 401:
                    raise RuntimeError(
                        f"OSF API 401 (unauthorized) for {url}. "
                        f"If the project is private, set env var {cls.OSF_TOKEN_ENV}."
                    )
                if resp.status_code == 429 or resp.status_code >= 500:
                    raise HTTPError(f"OSF API status {resp.status_code} for {url}")
                resp.raise_for_status()
                return resp.json()
            except (ConnectionError, Timeout, HTTPError, RequestException) as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                wait = (2**attempt) + random.uniform(0, 1)
                time.sleep(wait)
            except FileNotFoundError:
                raise
        raise RuntimeError(
            f"OSF API request failed after {max_retries} attempts: {url}"
        ) from last_exc

    @classmethod
    def _walk_node(cls, node_id: str, api_base: str, token: Optional[str]):
        """Yield {path, node_id, file_id, size} dicts for every file under
        the node's osfstorage root."""
        url = f"{api_base}/nodes/{quote(node_id, safe='')}/files/osfstorage/"
        yield from cls._walk_url(url, node_id, token)

    @classmethod
    def _walk_url(cls, url: str, node_id: str, token: Optional[str]):
        headers = {"Accept": "application/vnd.api+json"}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        next_url: Optional[str] = url + ("?" if "?" not in url else "&") + "page%5Bsize%5D=100"
        while next_url:
            payload = cls._get_with_retry(next_url, headers)

            for entry in payload.get("data", []):
                attrs = entry.get("attributes", {}) or {}
                kind = attrs.get("kind")
                rel_path = (attrs.get("materialized_path") or "").lstrip("/")
                if kind == "file":
                    yield {
                        "path": rel_path,
                        "node_id": node_id,
                        "file_id": entry.get("id"),
                        "size": attrs.get("size"),
                    }
                elif kind == "folder":
                    folder_url = (
                        ((entry.get("relationships") or {}).get("files") or {})
                        .get("links", {}).get("related", {}).get("href")
                    )
                    if folder_url:
                        yield from cls._walk_url(folder_url, node_id, token)

            next_url = (payload.get("links") or {}).get("next")

    # ------------------------------------------------------------------
    # Download with retry
    # ------------------------------------------------------------------

    @classmethod
    def _download_with_retry_static(
        cls,
        fpath: str,
        rel_path: str,
        entry: dict,
        files_base: str,
        token: Optional[str],
        max_retries: int = 5,
        timeout_download_s: int = 120,
    ) -> str:
        if entry is None:
            raise FileNotFoundError(
                f"OSF file not resolved before download: '{rel_path}'."
            )

        node_id = entry["node_id"]
        file_id = entry["file_id"]
        total_bytes = entry.get("size")
        download_url = (
            f"{files_base}/resources/{quote(node_id, safe='')}"
            f"/providers/osfstorage/{quote(file_id, safe='')}"
        )

        headers = {}
        if token:
            headers["Authorization"] = f"Bearer {token}"

        last_exc: Optional[BaseException] = None
        for attempt in range(1, max_retries + 1):
            try:
                tmp_path = fpath + ".tmp"
                if os.path.exists(tmp_path):
                    try:
                        os.remove(tmp_path)
                    except Exception:
                        pass

                with requests.get(
                    download_url, headers=headers, stream=True, timeout=timeout_download_s
                ) as r:
                    if r.status_code == 401:
                        raise RuntimeError(
                            f"Unauthorized downloading '{rel_path}' from OSF node "
                            f"'{node_id}'. If the project is private, set env var "
                            f"{cls.OSF_TOKEN_ENV}."
                        )
                    if r.status_code == 404:
                        raise FileNotFoundError(
                            f"OSF download URL 404: {download_url}"
                        )
                    if r.status_code == 429:
                        raise RuntimeError(
                            f"OSF rate limit exceeded downloading '{rel_path}'."
                        )
                    if r.status_code >= 500:
                        raise HTTPError(f"OSF server error {r.status_code}")
                    r.raise_for_status()

                    try:
                        clen = r.headers.get("Content-Length")
                        if clen:
                            total_bytes = int(clen)
                    except Exception:
                        pass

                    show_progress = cls._progress_enabled() and (
                        total_bytes is None or total_bytes >= 10 * 1024 * 1024
                    )
                    pbar = None
                    if show_progress:
                        try:
                            from tqdm.auto import tqdm  # type: ignore

                            desc = f"OSF {os.path.basename(fpath)}"
                            pbar = tqdm(
                                total=total_bytes,
                                unit="B",
                                unit_scale=True,
                                desc=desc,
                                leave=True,
                            )
                        except Exception:
                            pbar = None

                    downloaded = 0
                    start_t = time.monotonic()
                    last_report_t = start_t
                    last_len = 0

                    def _write_line(line: str, end: bool = False) -> None:
                        nonlocal last_len
                        pad = max(0, last_len - len(line))
                        with cls._print_lock:
                            sys.stderr.write("\r" + line + (" " * pad))
                            if end:
                                sys.stderr.write("\n")
                            sys.stderr.flush()
                        last_len = len(line)

                    def _make_line(*, final: bool = False) -> str:
                        name = os.path.basename(fpath)
                        dt = max(1e-6, time.monotonic() - start_t)
                        rate = downloaded / dt
                        if total_bytes:
                            pct = int((downloaded / max(total_bytes, 1)) * 100)
                            width = 24
                            filled = int((pct / 100.0) * width)
                            bar = ("#" * filled) + ("-" * (width - filled))
                            line = (
                                f"OSF: {name} [{bar}] {pct:3d}% "
                                f"({cls._format_bytes(downloaded)} / "
                                f"{cls._format_bytes(total_bytes)}) "
                                f"at {cls._format_bytes(int(rate))}/s"
                            )
                        else:
                            line = (
                                f"OSF: {name} {cls._format_bytes(downloaded)} "
                                f"at {cls._format_bytes(int(rate))}/s"
                            )
                        if final:
                            line += f" (done in {dt:.1f}s)"
                        return line

                    try:
                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                if not show_progress:
                                    continue
                                downloaded += len(chunk)
                                if pbar is not None:
                                    try:
                                        pbar.update(len(chunk))
                                    except Exception:
                                        pass
                                else:
                                    now = time.monotonic()
                                    if (now - last_report_t) >= 1.0:
                                        last_report_t = now
                                        _write_line(_make_line())

                        os.replace(tmp_path, fpath)
                        if show_progress:
                            if pbar is not None:
                                try:
                                    pbar.close()
                                except Exception:
                                    pass
                            else:
                                _write_line(_make_line(final=True), end=True)
                        return fpath
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass

            except (ConnectionError, Timeout, HTTPError, RequestException) as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                wait = (2**attempt) + random.uniform(0, 1)
                print(
                    f"Network/HTTP error downloading {os.path.basename(fpath)} from OSF, "
                    f"retrying in {wait:.1f}s ({attempt}/{max_retries})"
                )
                time.sleep(wait)
            except FileNotFoundError:
                raise
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "unauthorized" in msg or "401" in msg:
                    raise
                if attempt >= max_retries:
                    break
                wait = (2**attempt) + random.uniform(0, 1)
                print(
                    f"Error downloading {os.path.basename(fpath)} from OSF, "
                    f"retrying in {wait:.1f}s ({attempt}/{max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to download '{rel_path}' from OSF after {max_retries} attempts."
        ) from last_exc
