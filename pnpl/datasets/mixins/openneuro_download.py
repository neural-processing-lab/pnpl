"""
OpenNeuroDownloadMixin — download files from OpenNeuro datasets.

OpenNeuro (https://openneuro.org) hosts BIDS-formatted neuroimaging data
on a public AWS S3 bucket. Individual files are accessible via plain
HTTPS at:

    https://s3.amazonaws.com/openneuro.org/<dataset_id>/<bids-path>

so this mixin is much simpler than the OSF / Radboud counterparts: no
auth, no per-folder API walks, just GET-with-retry. We keep the same
public surface (``ensure_file`` / ``prefetch_files``) so dataset
classes can swap download backends without changing call sites.

For datasets whose BIDS axes (subjects / sessions / tasks / runs) are
known up front via the dataset's ``constants`` module, the mixin's
``list_remote_files`` is not needed at all — callers construct the
expected paths themselves and ``ensure_file`` fetches each on demand.
For ad-hoc enumeration we expose
:meth:`list_remote_files`, which queries OpenNeuro's GraphQL API.
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


_S3_BASE = "https://s3.amazonaws.com/openneuro.org"
_GRAPHQL_URL = "https://openneuro.org/crn/graphql"


class OpenNeuroDownloadMixin:
    """
    Mixin providing OpenNeuro download functionality.

    Classes using this mixin should define:

    - ``OPENNEURO_DATASET_ID``: str — e.g. ``"ds007523"``.

    Optional class-level overrides:

    - ``OPENNEURO_SNAPSHOT_TAG``: str — version tag, e.g. ``"1.0.1"``.
      Only used by :meth:`list_remote_files`; downloads themselves go
      against the bucket root and pick up whatever the ``HEAD`` of the
      bucket holds (OpenNeuro keeps every published version forever, so
      missing files only mean the path is wrong, not that the version
      moved).
    - ``OPENNEURO_S3_BASE``: str — defaults to
      ``https://s3.amazonaws.com/openneuro.org``.

    Expected instance attributes:

    - ``data_path``: str — local data directory; remote paths are
      mapped relative to this.
    - ``download``: bool — whether downloading is enabled.
    """

    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)
    _download_futures: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    _print_lock: ClassVar[threading.Lock] = threading.Lock()
    _file_listing_cache: ClassVar[dict] = {}

    OPENNEURO_DATASET_ID: ClassVar[str] = ""
    OPENNEURO_SNAPSHOT_TAG: ClassVar[Optional[str]] = None
    OPENNEURO_S3_BASE: ClassVar[str] = _S3_BASE

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
        v = os.getenv("PNPL_OPENNEURO_PROGRESS")
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
    def _dataset_id(cls) -> str:
        ds = (cls.OPENNEURO_DATASET_ID or "").strip()
        if not ds:
            raise RuntimeError(
                "OPENNEURO_DATASET_ID is not set on the dataset class; "
                "cannot download from OpenNeuro."
            )
        return ds

    @classmethod
    def _join_url(cls, rel_path: str) -> str:
        clean = rel_path.lstrip("/")
        if not clean:
            return f"{cls.OPENNEURO_S3_BASE.rstrip('/')}/{cls._dataset_id()}/"
        encoded = "/".join(quote(seg, safe="") for seg in clean.split("/"))
        return f"{cls.OPENNEURO_S3_BASE.rstrip('/')}/{cls._dataset_id()}/{encoded}"

    # ------------------------------------------------------------------
    # Public download surface
    # ------------------------------------------------------------------

    def prefetch_files(self, file_paths: list[str]) -> None:
        """Prefetch multiple files in parallel (skips already-present)."""
        seen: set[str] = set()
        targets: list[str] = []
        for fpath in file_paths:
            if fpath in seen or os.path.exists(fpath):
                continue
            seen.add(fpath)
            targets.append(fpath)
        if not targets:
            return

        n = len(targets)
        type(self)._log(f"OpenNeuro: downloading {n} file(s)…")

        scheduled = [(fpath, self._schedule_download(fpath)) for fpath in targets]

        completed = 0
        for fpath, future in scheduled:
            try:
                future.result()
                completed += 1
                size = os.path.getsize(fpath) if os.path.exists(fpath) else None
                size_str = type(self)._format_bytes(size)
                type(self)._log(
                    f"  [{completed}/{n}] {os.path.basename(fpath)} ({size_str})"
                )
            except Exception as exc:
                type(self)._log(
                    f"  [error] {os.path.basename(fpath)}: {exc}"
                )
        type(self)._log(f"OpenNeuro: done ({completed}/{n} files).")

    def ensure_file(self, fpath: str) -> str:
        """Ensure a file exists locally, downloading from OpenNeuro if needed."""
        if os.path.exists(fpath):
            return fpath
        if not getattr(self, "download", True):
            raise FileNotFoundError(
                f"File not found: {fpath}. Download is disabled."
            )
        future = self._schedule_download(fpath)
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
                    download_url=type(self)._join_url(rel_path),
                )
            return self._download_futures[fpath]

    # ------------------------------------------------------------------
    # Lazy single-file probe
    # ------------------------------------------------------------------

    @classmethod
    def resolve_remote_file(cls, rel_path: str) -> dict:
        """Return ``{"size", "url"}`` for a remote path via HEAD.

        Raises :class:`FileNotFoundError` if the path doesn't exist.
        """
        url = cls._join_url(rel_path)
        for attempt in range(1, 5):
            try:
                resp = requests.head(url, timeout=30, allow_redirects=True)
                if resp.status_code == 404:
                    raise FileNotFoundError(
                        f"OpenNeuro: not found '{rel_path}' ({url})"
                    )
                if resp.status_code == 403:
                    raise FileNotFoundError(
                        f"OpenNeuro: forbidden '{rel_path}' — likely missing "
                        f"({url})"
                    )
                resp.raise_for_status()
                size: Optional[int] = None
                try:
                    clen = resp.headers.get("Content-Length")
                    if clen:
                        size = int(clen)
                except Exception:
                    size = None
                return {"size": size, "url": url}
            except FileNotFoundError:
                raise
            except (ConnectionError, Timeout, HTTPError, RequestException):
                if attempt >= 4:
                    break
                time.sleep((2**attempt) + random.uniform(0, 1))
        raise RuntimeError(f"OpenNeuro HEAD failed after retries: {url}")

    # ------------------------------------------------------------------
    # GraphQL listing (optional)
    # ------------------------------------------------------------------

    @classmethod
    def list_remote_files(cls, refresh: bool = False) -> list[str]:
        """
        Return dataset-relative file paths advertised by OpenNeuro's
        GraphQL API for the configured snapshot.

        Requires ``OPENNEURO_SNAPSHOT_TAG`` to be set on the class.
        Datasets whose BIDS axes are known up front (via a ``constants``
        module listing subjects / sessions / tasks / runs) usually do
        not need this — :meth:`ensure_file` is enough on its own.
        """
        ds_id = cls._dataset_id()
        tag = cls.OPENNEURO_SNAPSHOT_TAG
        if not tag:
            raise RuntimeError(
                "OPENNEURO_SNAPSHOT_TAG must be set to use list_remote_files."
            )
        cache_key = (ds_id, tag)
        if not refresh and cache_key in cls._file_listing_cache:
            return cls._file_listing_cache[cache_key]

        cls._log(
            f"OpenNeuro: listing files for {ds_id}@{tag} via GraphQL…"
        )
        files: list[str] = []
        cls._collect_snapshot_files(ds_id, tag, tree=None, prefix="", out=files)
        files.sort()
        cls._file_listing_cache[cache_key] = files
        return files

    @classmethod
    def _collect_snapshot_files(
        cls,
        ds_id: str,
        tag: str,
        tree: Optional[str],
        prefix: str,
        out: list[str],
    ) -> None:
        if tree is None:
            query = (
                "query($id:ID!,$tag:String!){snapshot(datasetId:$id,tag:$tag)"
                "{files{id filename directory size}}}"
            )
            variables = {"id": ds_id, "tag": tag}
        else:
            query = (
                "query($id:ID!,$tag:String!,$tree:String!){"
                "snapshot(datasetId:$id,tag:$tag){"
                "files(tree:$tree){id filename directory size}}}"
            )
            variables = {"id": ds_id, "tag": tag, "tree": tree}

        payload = cls._graphql(query, variables)
        snapshot = (payload.get("data") or {}).get("snapshot") or {}
        entries = snapshot.get("files") or []

        for entry in entries:
            name = entry.get("filename")
            if not name:
                continue
            child_prefix = f"{prefix}{name}" if not prefix else f"{prefix}/{name}"
            if entry.get("directory"):
                cls._collect_snapshot_files(
                    ds_id, tag, tree=entry.get("id"),
                    prefix=child_prefix, out=out,
                )
            else:
                out.append(child_prefix)

    @classmethod
    def _graphql(cls, query: str, variables: dict) -> dict:
        last_exc: Optional[BaseException] = None
        for attempt in range(1, 5):
            try:
                resp = requests.post(
                    _GRAPHQL_URL,
                    json={"query": query, "variables": variables},
                    timeout=30,
                )
                if resp.status_code >= 500:
                    raise HTTPError(f"OpenNeuro GraphQL {resp.status_code}")
                resp.raise_for_status()
                payload = resp.json()
                if "errors" in payload and payload["errors"]:
                    raise RuntimeError(
                        f"OpenNeuro GraphQL errors: {payload['errors']}"
                    )
                return payload
            except (ConnectionError, Timeout, HTTPError, RequestException) as exc:
                last_exc = exc
                if attempt >= 4:
                    break
                time.sleep((2**attempt) + random.uniform(0, 1))
        raise RuntimeError(
            "OpenNeuro GraphQL request failed after retries."
        ) from last_exc

    # ------------------------------------------------------------------
    # Download with retry + progress
    # ------------------------------------------------------------------

    @classmethod
    def _download_with_retry_static(
        cls,
        fpath: str,
        rel_path: str,
        download_url: str,
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
                    download_url, stream=True, timeout=timeout_download_s,
                ) as r:
                    if r.status_code == 404 or r.status_code == 403:
                        # 403 from S3 typically means the key doesn't exist
                        # under this prefix (the bucket isn't anonymously
                        # listable). Treat as fail-fast not-found.
                        raise FileNotFoundError(
                            f"OpenNeuro {r.status_code}: {download_url}"
                        )
                    if r.status_code >= 500:
                        raise HTTPError(f"OpenNeuro server error {r.status_code}")
                    r.raise_for_status()

                    total_bytes: Optional[int] = None
                    try:
                        clen = r.headers.get("Content-Length")
                        if clen:
                            total_bytes = int(clen)
                    except Exception:
                        total_bytes = None

                    show_progress = cls._progress_enabled() and (
                        total_bytes is None or total_bytes >= 10 * 1024 * 1024
                    )
                    pbar = None
                    if show_progress:
                        try:
                            from tqdm.auto import tqdm  # type: ignore

                            pbar = tqdm(
                                total=total_bytes, unit="B", unit_scale=True,
                                desc=f"OpenNeuro {os.path.basename(fpath)}",
                                leave=True,
                            )
                        except Exception:
                            pbar = None

                    try:
                        with open(tmp_path, "wb") as f:
                            for chunk in r.iter_content(chunk_size=1024 * 1024):
                                if not chunk:
                                    continue
                                f.write(chunk)
                                if pbar is not None:
                                    try:
                                        pbar.update(len(chunk))
                                    except Exception:
                                        pass

                        os.replace(tmp_path, fpath)
                        if pbar is not None:
                            try:
                                pbar.close()
                            except Exception:
                                pass
                        return fpath
                    finally:
                        if os.path.exists(tmp_path):
                            try:
                                os.remove(tmp_path)
                            except Exception:
                                pass

            except FileNotFoundError:
                raise
            except (ConnectionError, Timeout, HTTPError, RequestException) as e:
                last_exc = e
                if attempt >= max_retries:
                    break
                wait = (2**attempt) + random.uniform(0, 1)
                cls._log(
                    f"OpenNeuro: network error on {os.path.basename(fpath)}, "
                    f"retrying in {wait:.1f}s ({attempt}/{max_retries})"
                )
                time.sleep(wait)

        raise RuntimeError(
            f"Failed to download '{rel_path}' from OpenNeuro after "
            f"{max_retries} attempts."
        ) from last_exc
