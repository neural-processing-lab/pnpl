"""
RadboudDownloadMixin — WebDAV download for Radboud Data Repository
datasets (https://webdav.data.ru.nl/dccn/...).

The Radboud Data Repository is *not* open-access. Access requires a
data-sharing agreement with the dataset owner; once approved, you log
in with a personal username (often an ORCID with the
``@orcid.org`` suffix) and password over HTTP Basic auth.

Mirrors the surface of :class:`OSFDownloadMixin` so a dataset can swap
download backends without changing call sites:

  - ``ensure_file(local_path)`` — download if missing, return local path.
  - ``prefetch_files(local_paths)`` — parallel batch download.
  - ``ensure_directory(local_dir)`` — recursively download a remote
    directory (used for CTF ``.ds`` "files", which are really directories).
  - ``resolve_remote_file(rel_path)`` — lazy single-file lookup by walking
    only the folders on its path.

Auth:
  - Env vars ``RADBOUD_USERNAME`` and ``RADBOUD_PASSWORD`` (default).
    Override the env-var names per-class via ``RADBOUD_USERNAME_ENV`` /
    ``RADBOUD_PASSWORD_ENV``.
  - Without credentials, the mixin raises a clear ``RuntimeError``
    pointing at the missing env vars.
"""

from __future__ import annotations

import os
import random
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar, Optional
from urllib.parse import quote, unquote, urlparse

import requests
from requests.auth import HTTPBasicAuth
from requests.exceptions import ConnectionError, HTTPError, RequestException, Timeout


_DAV_NS = "{DAV:}"


class RadboudDownloadMixin:
    """
    Mixin providing WebDAV-backed download from the Radboud Data
    Repository.

    Classes using this mixin should define:
    - ``RADBOUD_DATASET_URL``: str — full WebDAV URL of the dataset root,
      e.g. ``"https://webdav.data.ru.nl/dccn/DSC_3011085.05_995_v1/"``.

    Optional class-level overrides:
    - ``RADBOUD_USERNAME_ENV`` (default ``"RADBOUD_USERNAME"``)
    - ``RADBOUD_PASSWORD_ENV`` (default ``"RADBOUD_PASSWORD"``)

    Expected instance attributes:
    - ``data_path``: str — local directory; remote paths are mapped
      relative to this.
    - ``download``: bool — whether downloading is enabled.
    """

    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)
    _download_futures: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _folder_listing_cache: ClassVar[dict] = {}
    _print_lock: ClassVar[threading.Lock] = threading.Lock()

    RADBOUD_DATASET_URL: ClassVar[str] = ""
    RADBOUD_USERNAME_ENV: ClassVar[str] = "RADBOUD_USERNAME"
    RADBOUD_PASSWORD_ENV: ClassVar[str] = "RADBOUD_PASSWORD"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _is_truthy(v: Optional[str]) -> bool:
        if v is None:
            return False
        return v.strip().lower() in {"1", "true", "yes", "y", "on"}

    @classmethod
    def _progress_enabled(cls) -> bool:
        v = os.getenv("PNPL_RADBOUD_PROGRESS")
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
    def _log(cls, message: str) -> None:
        if not cls._progress_enabled():
            return
        with cls._print_lock:
            sys.stderr.write(message + "\n")
            sys.stderr.flush()

    @classmethod
    def _credentials(cls) -> Optional[HTTPBasicAuth]:
        user = os.getenv(cls.RADBOUD_USERNAME_ENV)
        pwd = os.getenv(cls.RADBOUD_PASSWORD_ENV)
        if user and pwd:
            return HTTPBasicAuth(user, pwd)
        return None

    @classmethod
    def _require_credentials(cls) -> HTTPBasicAuth:
        creds = cls._credentials()
        if creds is None:
            raise RuntimeError(
                f"Radboud Data Repository credentials are required. "
                f"Set env vars {cls.RADBOUD_USERNAME_ENV} and "
                f"{cls.RADBOUD_PASSWORD_ENV} (e.g., your ORCID-suffixed "
                f"username and the password from the data-sharing portal)."
            )
        return creds

    @classmethod
    def _dataset_url(cls) -> str:
        url = (cls.RADBOUD_DATASET_URL or "").rstrip("/")
        if not url:
            raise RuntimeError(
                "RADBOUD_DATASET_URL is not set on the dataset class; "
                "cannot download from the Radboud Data Repository."
            )
        return url + "/"

    @classmethod
    def _join_url(cls, rel_path: str) -> str:
        # Each path segment is URL-encoded individually so spaces and
        # other special characters survive intact.
        clean = rel_path.lstrip("/")
        if not clean:
            return cls._dataset_url()
        encoded = "/".join(quote(seg, safe="") for seg in clean.split("/"))
        return cls._dataset_url() + encoded

    # ------------------------------------------------------------------
    # Public download surface
    # ------------------------------------------------------------------

    def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files in parallel (skips already-present)."""
        futures = []
        for fpath in {p for p in file_paths if not os.path.exists(p)}:
            futures.append(self._schedule_download(fpath))
        if futures:
            print(f"Downloading {len(futures)} files from Radboud WebDAV...")
            for future in futures:
                try:
                    future.result()
                except Exception as e:
                    print(f"Error downloading a file: {e}")
            print("Done!")

    def ensure_file(self, fpath: str) -> str:
        """Ensure a file exists locally, downloading via WebDAV if needed."""
        if os.path.exists(fpath):
            return fpath
        if not getattr(self, "download", True):
            raise FileNotFoundError(
                f"File not found: {fpath}. Download is disabled."
            )
        future = self._schedule_download(fpath)
        return future.result()

    def ensure_directory(self, dpath: str) -> str:
        """Recursively download a remote directory to ``dpath``.

        Used for CTF ``.ds`` "files" which are really directories of
        binary chunks (``.meg4`` data, ``.res4`` header, etc.). All
        contained files are downloaded; the directory is returned.
        """
        if os.path.exists(dpath) and os.listdir(dpath):
            return dpath
        if not getattr(self, "download", True):
            raise FileNotFoundError(
                f"Directory not found: {dpath}. Download is disabled."
            )

        rel_path = os.path.relpath(dpath, self.data_path).replace(os.path.sep, "/")
        files = type(self)._list_files_recursive(rel_path)
        os.makedirs(dpath, exist_ok=True)

        if not files:
            raise FileNotFoundError(
                f"No files found at remote directory '{rel_path}'."
            )

        # Schedule each file for download; rel paths inside the directory
        # are resolved against ``self.data_path`` like any other ensure_file.
        local_targets = []
        for entry in files:
            local_target = os.path.join(self.data_path, *entry["rel_path"].split("/"))
            os.makedirs(os.path.dirname(local_target), exist_ok=True)
            local_targets.append(local_target)
        self.prefetch_files(local_targets)
        return dpath

    def _schedule_download(self, fpath: str):
        rel_path = os.path.relpath(fpath, self.data_path).replace(os.path.sep, "/")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)

        with self._lock:
            if fpath not in self._download_futures:
                # Resolve size lazily so we have a content length up front
                # for the progress bar (requests' Content-Length header is
                # also a fallback inside the worker).
                size = type(self)._lookup_remote_size(rel_path)
                self._download_futures[fpath] = self._executor.submit(
                    self._download_with_retry_static,
                    fpath=fpath,
                    rel_path=rel_path,
                    download_url=type(self)._join_url(rel_path),
                    auth=type(self)._require_credentials(),
                    expected_size=size,
                )
            return self._download_futures[fpath]

    # ------------------------------------------------------------------
    # Lazy folder/file resolution (PROPFIND)
    # ------------------------------------------------------------------

    @classmethod
    def resolve_remote_file(cls, rel_path: str) -> dict:
        """Return ``{"size", "is_collection", "url"}`` for a remote path,
        without listing siblings or descendants.

        Raises :class:`FileNotFoundError` if the path doesn't exist.
        """
        return cls._stat(rel_path)

    @classmethod
    def _stat(cls, rel_path: str) -> dict:
        url = cls._join_url(rel_path)
        auth = cls._require_credentials()
        # Depth: 0 → just the resource itself.
        responses = cls._propfind(url, auth, depth="0")
        if not responses:
            raise FileNotFoundError(f"Radboud WebDAV: not found '{rel_path}'.")
        return responses[0]

    @classmethod
    def _list_folder(cls, rel_path: str) -> list[dict]:
        """List immediate children of a remote directory. Cached per
        (dataset_url, rel_path)."""
        cache_key = (cls._dataset_url(), rel_path.strip("/"))
        cached = cls._folder_listing_cache.get(cache_key)
        if cached is not None:
            return cached

        url = cls._join_url(rel_path)
        if not url.endswith("/"):
            url += "/"
        auth = cls._require_credentials()
        responses = cls._propfind(url, auth, depth="1")

        # Drop the entry that refers to the directory itself.
        prefix = urlparse(url).path
        children = [r for r in responses if r["url_path"].rstrip("/") != prefix.rstrip("/")]
        cls._folder_listing_cache[cache_key] = children
        return children

    @classmethod
    def _list_files_recursive(cls, rel_path: str) -> list[dict]:
        """All files under ``rel_path`` (recursive). Used by
        ``ensure_directory``."""
        files: list[dict] = []
        for entry in cls._list_folder(rel_path):
            child_rel = entry["rel_path"]
            if entry["is_collection"]:
                files.extend(cls._list_files_recursive(child_rel))
            else:
                files.append(entry)
        return files

    @classmethod
    def _lookup_remote_size(cls, rel_path: str) -> Optional[int]:
        """Best-effort size lookup. Falls back to None — the download
        worker also reads Content-Length."""
        try:
            stat = cls._stat(rel_path)
        except (FileNotFoundError, RuntimeError):
            return None
        size = stat.get("size")
        try:
            return int(size) if size is not None else None
        except (TypeError, ValueError):
            return None

    @classmethod
    def _propfind(cls, url: str, auth: HTTPBasicAuth, depth: str) -> list[dict]:
        """Issue a PROPFIND and parse the multistatus response into a
        list of {url_path, rel_path, is_collection, size}."""
        body = (
            '<?xml version="1.0" encoding="utf-8"?>'
            '<propfind xmlns="DAV:"><prop>'
            '<resourcetype/><getcontentlength/>'
            '</prop></propfind>'
        )
        headers = {
            "Depth": depth,
            "Content-Type": "application/xml; charset=utf-8",
        }
        last_exc: Optional[BaseException] = None
        for attempt in range(1, 5):
            try:
                resp = requests.request(
                    "PROPFIND", url,
                    headers=headers, data=body, auth=auth, timeout=30,
                )
                if resp.status_code == 401:
                    raise RuntimeError(
                        f"Radboud WebDAV 401 (unauthorized) for {url}. "
                        f"Check {cls.RADBOUD_USERNAME_ENV} / "
                        f"{cls.RADBOUD_PASSWORD_ENV}."
                    )
                if resp.status_code == 404:
                    raise FileNotFoundError(
                        f"Radboud WebDAV 404: {url}"
                    )
                if resp.status_code == 207:
                    return cls._parse_multistatus(resp.text)
                if resp.status_code >= 500:
                    raise HTTPError(f"WebDAV {resp.status_code} for {url}")
                resp.raise_for_status()
                return cls._parse_multistatus(resp.text)
            except (FileNotFoundError, RuntimeError):
                raise
            except (ConnectionError, Timeout, HTTPError, RequestException) as exc:
                last_exc = exc
                if attempt >= 4:
                    break
                time.sleep((2**attempt) + random.uniform(0, 1))
        raise RuntimeError(
            f"Radboud WebDAV PROPFIND failed after retries: {url}"
        ) from last_exc

    @classmethod
    def _parse_multistatus(cls, xml_text: str) -> list[dict]:
        root = ET.fromstring(xml_text)
        dataset_path = urlparse(cls._dataset_url()).path
        results = []
        for response in root.findall(f"{_DAV_NS}response"):
            href_el = response.find(f"{_DAV_NS}href")
            if href_el is None or href_el.text is None:
                continue
            href = href_el.text
            url_path = unquote(href)

            is_collection = False
            size: Optional[int] = None
            for propstat in response.findall(f"{_DAV_NS}propstat"):
                status_el = propstat.find(f"{_DAV_NS}status")
                if status_el is not None and "200" not in (status_el.text or ""):
                    continue
                prop = propstat.find(f"{_DAV_NS}prop")
                if prop is None:
                    continue
                rt = prop.find(f"{_DAV_NS}resourcetype")
                if rt is not None and rt.find(f"{_DAV_NS}collection") is not None:
                    is_collection = True
                cl = prop.find(f"{_DAV_NS}getcontentlength")
                if cl is not None and cl.text:
                    try:
                        size = int(cl.text)
                    except ValueError:
                        pass

            # Convert absolute server path to dataset-relative.
            rel = url_path
            if rel.startswith(dataset_path):
                rel = rel[len(dataset_path):]
            rel = rel.strip("/")
            results.append({
                "url_path": url_path,
                "rel_path": rel,
                "is_collection": is_collection,
                "size": size,
            })
        return results

    # ------------------------------------------------------------------
    # File download
    # ------------------------------------------------------------------

    @classmethod
    def _download_with_retry_static(
        cls,
        fpath: str,
        rel_path: str,
        download_url: str,
        auth: HTTPBasicAuth,
        expected_size: Optional[int] = None,
        max_retries: int = 5,
        timeout_download_s: int = 120,
    ) -> str:
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
                    download_url, auth=auth, stream=True,
                    timeout=timeout_download_s,
                ) as r:
                    if r.status_code == 401:
                        raise RuntimeError(
                            f"Radboud WebDAV 401 downloading '{rel_path}'. "
                            f"Check {cls.RADBOUD_USERNAME_ENV} / "
                            f"{cls.RADBOUD_PASSWORD_ENV}."
                        )
                    if r.status_code == 404:
                        raise FileNotFoundError(f"Radboud WebDAV 404: {download_url}")
                    if r.status_code >= 500:
                        raise HTTPError(f"WebDAV server error {r.status_code}")
                    r.raise_for_status()

                    total_bytes = expected_size
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

                            pbar = tqdm(
                                total=total_bytes, unit="B", unit_scale=True,
                                desc=f"Radboud {os.path.basename(fpath)}",
                                leave=True,
                            )
                        except Exception:
                            pbar = None

                    downloaded = 0
                    start_t = time.monotonic()
                    last_report = start_t
                    last_len = 0

                    def _line() -> str:
                        name = os.path.basename(fpath)
                        dt = max(1e-6, time.monotonic() - start_t)
                        rate = downloaded / dt
                        if total_bytes:
                            pct = int((downloaded / max(total_bytes, 1)) * 100)
                            return (
                                f"Radboud: {name} {pct:3d}% "
                                f"({cls._format_bytes(downloaded)} / "
                                f"{cls._format_bytes(total_bytes)}) "
                                f"at {cls._format_bytes(int(rate))}/s"
                            )
                        return (
                            f"Radboud: {name} {cls._format_bytes(downloaded)} "
                            f"at {cls._format_bytes(int(rate))}/s"
                        )

                    def _write(line: str, end: bool = False) -> None:
                        nonlocal last_len
                        pad = max(0, last_len - len(line))
                        with cls._print_lock:
                            sys.stderr.write("\r" + line + (" " * pad))
                            if end:
                                sys.stderr.write("\n")
                            sys.stderr.flush()
                        last_len = len(line)

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
                                    if now - last_report >= 1.0:
                                        last_report = now
                                        _write(_line())

                        os.replace(tmp_path, fpath)
                        if show_progress:
                            if pbar is not None:
                                try:
                                    pbar.close()
                                except Exception:
                                    pass
                            else:
                                _write(_line(), end=True)
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
                    f"Network/HTTP error downloading {os.path.basename(fpath)} "
                    f"from Radboud WebDAV, retrying in {wait:.1f}s "
                    f"({attempt}/{max_retries})"
                )
                time.sleep(wait)
            except FileNotFoundError:
                raise
            except Exception as e:
                last_exc = e
                msg = str(e).lower()
                if "401" in msg or "unauthorized" in msg:
                    raise
                if attempt >= max_retries:
                    break
                time.sleep((2**attempt) + random.uniform(0, 1))

        raise RuntimeError(
            f"Failed to download '{rel_path}' from Radboud WebDAV "
            f"after {max_retries} attempts."
        ) from last_exc
