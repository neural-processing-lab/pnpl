"""
OhanaDownloadMixin - Provides OHANA download functionality with retry logic.

OHANA is PNPL's dataset platform (served by `dataset-manager`).

This mixin downloads individual files on demand via the OHANA API:
- Fetch presigned download URL for a file: GET /api/download/{datasetSlug}/{filePath}
- Download the actual file from the presigned URL

Auth:
- Private datasets require an API key.
- Expected env var: OHANA_API_KEY (fallback: API_KEY)
- Authorization header: "Bearer <key>"
"""

from __future__ import annotations

import json
import os
import random
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Optional
from urllib.parse import quote
import sys

import requests
from requests.exceptions import ConnectionError, Timeout, HTTPError, RequestException


class OhanaDownloadMixin:
    """
    Mixin providing OHANA download functionality.

    Classes using this mixin should define:
    - OHANA_DATASET_SLUG: str - dataset slug in OHANA (e.g. "megnist")

    Optional overrides:
    - OHANA_BASE_URL: str - base URL (default: https://ohana.neuralprocessinglab.com)
    - OHANA_API_KEY_ENV: str - primary env var name (default: "OHANA_API_KEY")

    Expected instance attributes:
    - data_path: str - Local data directory
    - download: bool - Whether downloading is enabled
    """

    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)
    _download_futures: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _manifest_cache: ClassVar[dict] = {}

    OHANA_BASE_URL: ClassVar[str] = "https://ohana.neuralprocessinglab.com"
    OHANA_DATASET_SLUG: ClassVar[str] = ""
    OHANA_API_KEY_ENV: ClassVar[str] = "OHANA_API_KEY"
    _print_lock: ClassVar[threading.Lock] = threading.Lock()

    @staticmethod
    def _is_truthy(v: Optional[str]) -> bool:
        if v is None:
            return False
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def _progress_enabled(cls) -> bool:
        # Default on. Users can disable via PNPL_OHANA_PROGRESS=0
        v = os.getenv("PNPL_OHANA_PROGRESS")
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


    def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files in parallel (downloads only missing files)."""
        futures = []
        needed_files = set()

        for fpath in file_paths:
            if not os.path.exists(fpath):
                needed_files.add(fpath)

        for fpath in needed_files:
            futures.append(self._schedule_download(fpath))

        if futures:
            print(f"Downloading {len(futures)} files...")
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading a file: {e}")
            print("Done!")

    def ensure_file(self, fpath: str) -> str:
        """
        Ensure a file exists locally, downloading from OHANA if needed.

        Raises:
            FileNotFoundError: if file doesn't exist and download is disabled
            RuntimeError: if download is unauthorized (missing/invalid key), or download fails
        """
        if os.path.exists(fpath):
            return fpath

        if not getattr(self, "download", True):
            raise FileNotFoundError(f"File not found: {fpath}. Download is disabled.")

        future = self._schedule_download(fpath)
        return future.result()

    @classmethod
    def ensure_file_download(cls, fpath: str, data_path: str) -> str:
        """
        Class method to download a file without requiring dataset instantiation.
        """
        if os.path.exists(fpath):
            return fpath

        rel_path = os.path.relpath(fpath, data_path).replace(os.path.sep, "/")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with cls._lock:
            if fpath not in cls._download_futures:
                cls._download_futures[fpath] = cls._executor.submit(
                    cls._download_with_retry_static,
                    fpath=fpath,
                    rel_path=rel_path,
                    data_path=data_path,
                    base_url=cls._resolve_base_url(),
                    dataset_slug=cls.OHANA_DATASET_SLUG,
                    api_key_env=cls.OHANA_API_KEY_ENV,
                )
            future = cls._download_futures[fpath]

        return future.result()

    def _schedule_download(self, fpath: str):
        rel_path = os.path.relpath(fpath, self.data_path).replace(os.path.sep, "/")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with self._lock:
            if fpath not in self._download_futures:
                self._download_futures[fpath] = self._executor.submit(
                    self._download_with_retry_static,
                    fpath=fpath,
                    rel_path=rel_path,
                    data_path=self.data_path,
                    base_url=self._resolve_base_url(),
                    dataset_slug=self.OHANA_DATASET_SLUG,
                    api_key_env=getattr(self, "OHANA_API_KEY_ENV", "OHANA_API_KEY"),
                )
            return self._download_futures[fpath]

    @classmethod
    def _resolve_base_url(cls) -> str:
        # Allow overriding globally via env var (useful for staging/dev).
        return (os.getenv("OHANA_BASE_URL") or cls.OHANA_BASE_URL or "").rstrip("/")

    @classmethod
    def get_dataset_manifest(cls, refresh: bool = False) -> dict:
        """
        Fetch dataset metadata and file manifest from OHANA.

        Returns the JSON payload from ``GET /api/datasets/{slug}``.
        """
        dataset_slug = cls.OHANA_DATASET_SLUG
        if not dataset_slug:
            raise RuntimeError(
                "OHANA_DATASET_SLUG is not set on the dataset class; cannot fetch OHANA manifest."
            )

        base_url = cls._resolve_base_url()
        api_key_env = getattr(cls, "OHANA_API_KEY_ENV", "OHANA_API_KEY")
        api_key = cls._get_api_key(api_key_env)
        cache_key = (base_url, dataset_slug, api_key or "")

        if not refresh and cache_key in cls._manifest_cache:
            return cls._manifest_cache[cache_key]

        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        manifest_url = f"{base_url}/api/datasets/{quote(dataset_slug, safe='')}"
        resp = requests.get(manifest_url, headers=headers, timeout=30)

        if resp.status_code == 401:
            if api_key:
                raise RuntimeError(
                    f"Unauthorized fetching OHANA manifest for '{dataset_slug}' with the provided API key."
                )
            raise RuntimeError(
                f"Unauthorized fetching OHANA manifest for '{dataset_slug}'. "
                f"Set env var {api_key_env} (e.g. 'ohana_...') to access this private dataset."
            )
        if resp.status_code == 404:
            msg = cls._safe_json(resp).get("error") or "Dataset not found"
            raise FileNotFoundError(f"OHANA dataset manifest not found for '{dataset_slug}' ({msg}).")
        if resp.status_code == 429:
            msg = cls._safe_json(resp).get("error") or "Rate limit exceeded"
            raise RuntimeError(f"OHANA rate limit exceeded while fetching '{dataset_slug}' manifest: {msg}")
        if resp.status_code >= 500:
            raise RuntimeError(
                f"OHANA server error {resp.status_code} while fetching manifest for '{dataset_slug}'."
            )

        resp.raise_for_status()
        payload = resp.json()
        cls._manifest_cache[cache_key] = payload
        return payload

    @classmethod
    def list_remote_files(cls, refresh: bool = False) -> list[str]:
        """
        Return dataset-relative file paths advertised by the OHANA manifest.
        """
        payload = cls.get_dataset_manifest(refresh=refresh)
        dataset = payload.get("dataset") or {}
        files = dataset.get("files") or []
        paths = []
        for file_info in files:
            path = file_info.get("path")
            if isinstance(path, str) and path:
                paths.append(path)
        return paths

    @staticmethod
    def _get_api_key(api_key_env: str) -> Optional[str]:
        return os.getenv(api_key_env) or os.getenv("API_KEY")

    @staticmethod
    def _safe_json(resp: requests.Response) -> dict:
        try:
            return resp.json()
        except Exception:
            return {}

    @classmethod
    def _download_with_retry_static(
        cls,
        fpath: str,
        rel_path: str,
        data_path: str,
        base_url: str,
        dataset_slug: str,
        api_key_env: str = "OHANA_API_KEY",
        max_retries: int = 5,
        timeout_meta_s: int = 30,
        timeout_download_s: int = 120,
    ) -> str:
        if not dataset_slug:
            raise RuntimeError(
                "OHANA_DATASET_SLUG is not set on the dataset class; cannot download from OHANA."
            )

        api_key = cls._get_api_key(api_key_env)
        headers = {"Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        meta_url = (
            f"{base_url}/api/download/"
            f"{quote(dataset_slug, safe='')}/"
            f"{quote(rel_path, safe='/')}"
        )
        last_exc: Optional[BaseException] = None

        for attempt in range(1, max_retries + 1):
            try:
                meta_resp = requests.get(meta_url, headers=headers, timeout=timeout_meta_s)
                if meta_resp.status_code == 401:
                    if api_key:
                        raise RuntimeError(
                            f"Unauthorized downloading '{dataset_slug}/{rel_path}' from OHANA with provided API key. "
                            f"The key may be invalid or lacks access."
                        )
                    raise RuntimeError(
                        f"Unauthorized downloading '{dataset_slug}/{rel_path}' from OHANA. "
                        f"Set env var {api_key_env} (e.g. 'ohana_...') to access this private dataset."
                    )
                if meta_resp.status_code == 404:
                    msg = cls._safe_json(meta_resp).get("error") or "File not found"
                    raise FileNotFoundError(
                        f"OHANA file not found: '{dataset_slug}/{rel_path}' ({msg})."
                    )
                if meta_resp.status_code == 429:
                    msg = cls._safe_json(meta_resp).get("error") or "Rate limit exceeded"
                    raise RuntimeError(f"OHANA rate limit exceeded: {msg}")
                if meta_resp.status_code >= 500:
                    raise HTTPError(f"OHANA server error {meta_resp.status_code}")
                meta_resp.raise_for_status()

                meta = meta_resp.json()
                download_url = meta.get("downloadUrl")
                if not download_url:
                    raise RuntimeError(
                        f"OHANA response missing downloadUrl for '{dataset_slug}/{rel_path}'. "
                        f"Response: {json.dumps(meta)[:500]}"
                    )

                total_bytes: Optional[int] = None
                try:
                    # OHANA returns size as a string in JSON.
                    if "size" in meta:
                        total_bytes = int(meta["size"])
                except Exception:
                    total_bytes = None

                tmp_path = fpath + ".tmp"
                try:
                    # Atomic write: stream to a temporary file then rename into place.
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)

                    with requests.get(download_url, stream=True, timeout=timeout_download_s) as r:
                        if r.status_code >= 500:
                            raise HTTPError(f"Presigned download server error {r.status_code}")
                        r.raise_for_status()

                        # Prefer Content-Length if present.
                        try:
                            clen = r.headers.get("Content-Length")
                            if clen:
                                total_bytes = int(clen)
                        except Exception:
                            pass

                        show_progress = cls._progress_enabled() and (
                            total_bytes is None or total_bytes >= 10 * 1024 * 1024
                        )
                        downloaded = 0
                        start_t = time.monotonic()
                        last_report_t = start_t
                        last_len = 0

                        # Prefer a real progress bar in notebooks/Colab when available.
                        pbar = None
                        if show_progress:
                            try:  # optional dependency
                                from tqdm.auto import tqdm  # type: ignore

                                desc = f"OHANA {os.path.basename(fpath)}"
                                if total_bytes:
                                    pbar = tqdm(total=total_bytes, unit="B", unit_scale=True, desc=desc, leave=True)
                                else:
                                    # Unknown total: still show a bar that grows.
                                    pbar = tqdm(total=None, unit="B", unit_scale=True, desc=desc, leave=True)
                            except Exception:
                                pbar = None

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
                            # Single-line progress (fallback when tqdm isn't available).
                            name = os.path.basename(fpath)
                            dt = max(1e-6, time.monotonic() - start_t)
                            rate = downloaded / dt
                            if total_bytes:
                                pct = int((downloaded / max(total_bytes, 1)) * 100)
                                width = 24
                                filled = int((pct / 100.0) * width)
                                bar = ("#" * filled) + ("-" * (width - filled))
                                line = (
                                    f"OHANA: {name} [{bar}] {pct:3d}% "
                                    f"({cls._format_bytes(downloaded)} / {cls._format_bytes(total_bytes)}) "
                                    f"at {cls._format_bytes(int(rate))}/s"
                                )
                            else:
                                line = (
                                    f"OHANA: {name} {cls._format_bytes(downloaded)} "
                                    f"at {cls._format_bytes(int(rate))}/s"
                                )
                            if final:
                                line += f" (done in {dt:.1f}s)"
                            return line

                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if chunk:
                                    f.write(chunk)
                                    if show_progress:
                                        downloaded += len(chunk)
                                        if pbar is not None:
                                            try:
                                                pbar.update(len(chunk))
                                            except Exception:
                                                pass
                                        else:
                                            now = time.monotonic()
                                            # Throttle updates to avoid log spam.
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
                            # Ensure we print a final line and end with a newline.
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
                    f"Network/HTTP error downloading {os.path.basename(fpath)} from OHANA, "
                    f"retrying in {wait:.1f}s ({attempt}/{max_retries})"
                )
                time.sleep(wait)
            except Exception as e:
                last_exc = e
                # For auth/missing key/not found: don't retry.
                if isinstance(e, (FileNotFoundError,)):
                    raise
                msg = str(e).lower()
                if "unauthorized" in msg or "missing ohana api key" in msg:
                    raise
                if attempt >= max_retries:
                    break
                wait = (2**attempt) + random.uniform(0, 1)
                print(
                    f"Error downloading {os.path.basename(fpath)} from OHANA, "
                    f"retrying in {wait:.1f}s ({attempt}/{max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to download '{dataset_slug}/{rel_path}' from OHANA after {max_retries} attempts."
        ) from last_exc
