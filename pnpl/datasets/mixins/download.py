"""
HFDownloadMixin - Provides HuggingFace download functionality with retry logic.

This mixin handles downloading files from HuggingFace Hub with:
- Retry logic with exponential backoff
- Multiple repository fallbacks
- Thread-based parallel downloading
- Environment variable support (HF_TOKEN, HF_DATASET)
"""

import os
import time
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import ClassVar

from requests.exceptions import ConnectionError, Timeout, HTTPError


class HFDownloadMixin:
    """
    Mixin providing HuggingFace download functionality.
    
    Classes using this mixin should define:
    - HUGGINGFACE_REPO: str - Primary HuggingFace repository ID
    - HUGGINGFACE_FALLBACK_REPOS: list[str] - Optional fallback repositories
    - data_path: str - Local data directory
    - download: bool - Whether downloading is enabled
    """
    
    # Class-level thread pool and download tracking
    _executor: ClassVar[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=4)
    _download_futures: ClassVar[dict] = {}
    _lock: ClassVar[threading.Lock] = threading.Lock()
    
    # To be overridden by subclasses
    HUGGINGFACE_REPO: ClassVar[str] = ""
    HUGGINGFACE_FALLBACK_REPOS: ClassVar[list[str]] = []
    
    def prefetch_files(self, file_paths: list[str]) -> None:
        """
        Prefetch multiple files in parallel.
        
        Args:
            file_paths: List of local file paths to ensure exist
        """
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
    
    def _schedule_download(self, fpath: str):
        """Schedule a file download with retry logic."""
        rel_path = os.path.relpath(fpath, self.data_path)
        # Windows fix: convert Windows path separator to URL path separator
        rel_path = rel_path.replace(os.path.sep, '/')
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        
        with self._lock:
            if fpath not in self._download_futures:
                self._download_futures[fpath] = self._executor.submit(
                    self._download_with_retry,
                    fpath=fpath,
                    rel_path=rel_path,
                )
            return self._download_futures[fpath]
    
    def ensure_file(self, fpath: str) -> str:
        """
        Ensure a file exists locally, downloading if needed.
        
        Args:
            fpath: Full local path where file should exist
            
        Returns:
            Path to the file
            
        Raises:
            FileNotFoundError: If file doesn't exist and download is disabled
        """
        if os.path.exists(fpath):
            return fpath
        
        if not getattr(self, 'download', True):
            raise FileNotFoundError(f"File not found: {fpath}. Download is disabled.")
        
        future = self._schedule_download(fpath)
        return future.result()
    
    @classmethod
    def ensure_file_download(cls, fpath: str, data_path: str, repo_id: str = None) -> str:
        """
        Class method to download a file without requiring dataset instantiation.
        
        Args:
            fpath: Full path to the file that should exist locally
            data_path: Base data directory for computing relative paths
            repo_id: Optional specific repository to use
            
        Returns:
            Path to the downloaded file
        """
        if os.path.exists(fpath):
            return fpath
        
        rel_path = os.path.relpath(fpath, data_path)
        rel_path = rel_path.replace(os.path.sep, '/')
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        
        with cls._lock:
            if fpath not in cls._download_futures:
                cls._download_futures[fpath] = cls._executor.submit(
                    cls._download_with_retry_static,
                    fpath=fpath,
                    rel_path=rel_path,
                    data_path=data_path,
                    primary_repo=repo_id or cls.HUGGINGFACE_REPO,
                    fallback_repos=cls.HUGGINGFACE_FALLBACK_REPOS,
                )
            future = cls._download_futures[fpath]
        
        return future.result()
    
    def _download_with_retry(self, fpath: str, rel_path: str, max_retries: int = 5) -> str:
        """Instance method wrapper for download with retry."""
        return self._download_with_retry_static(
            fpath=fpath,
            rel_path=rel_path,
            data_path=self.data_path,
            primary_repo=self.HUGGINGFACE_REPO,
            fallback_repos=self.HUGGINGFACE_FALLBACK_REPOS,
            max_retries=max_retries,
        )
    
    @staticmethod
    def _download_with_retry_static(
        fpath: str,
        rel_path: str,
        data_path: str,
        primary_repo: str,
        fallback_repos: list[str] = None,
        max_retries: int = 5,
    ) -> str:
        """
        Download a file with retry logic and multiple repository fallback.
        
        Tries repositories in this order:
        1. Custom dataset from HF_DATASET env var (if HF_TOKEN is provided)
        2. Primary repository
        3. Fallback repositories
        """
        try:
            from huggingface_hub import hf_hub_download  # type: ignore
            from huggingface_hub.errors import (  # type: ignore
                RepositoryNotFoundError,
                EntryNotFoundError,
                GatedRepoError,
            )
        except Exception as e:  # pragma: no cover - depends on env install
            raise RuntimeError(
                "huggingface_hub is required for HuggingFace downloads. "
                "Install it (e.g. `pip install huggingface_hub`) or disable downloads."
            ) from e

        hf_token = os.getenv('HF_TOKEN')
        hf_dataset = os.getenv('HF_DATASET')
        
        repos_to_try = []
        
        # Try custom dataset first if both token and dataset are provided
        if hf_token and hf_dataset:
            print(f"Using custom dataset: {hf_dataset} (with token)")
            repos_to_try.append({"repo_id": hf_dataset, "token": hf_token})
        
        # Add primary repo
        if primary_repo:
            repos_to_try.append({"repo_id": primary_repo, "token": None})
        
        # Add fallback repos
        for repo in (fallback_repos or []):
            repos_to_try.append({"repo_id": repo, "token": None})
        
        last_exception = None
        
        for repo_config in repos_to_try:
            retries = 0
            while retries < max_retries:
                try:
                    download_kwargs = {
                        "repo_id": repo_config["repo_id"],
                        "repo_type": "dataset",
                        "filename": rel_path,
                        "local_dir": data_path,
                    }
                    
                    if repo_config["token"]:
                        download_kwargs["token"] = repo_config["token"]
                    
                    return hf_hub_download(**download_kwargs)
                    
                except (RepositoryNotFoundError, EntryNotFoundError, GatedRepoError) as e:
                    last_exception = e
                    break  # Move to next repository
                    
                except (ConnectionError, Timeout, HTTPError) as e:
                    last_exception = e
                    retries += 1
                    wait_time = 2 ** retries + random.uniform(0, 1)
                    if retries < max_retries:
                        print(f"Network error for {os.path.basename(fpath)} from {repo_config['repo_id']}, "
                              f"retrying in {wait_time:.1f}s ({retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        break
                        
                except Exception as e:
                    last_exception = e
                    retries += 1
                    wait_time = 2 ** retries + random.uniform(0, 1)
                    if retries < max_retries:
                        print(f"Unknown error for {os.path.basename(fpath)}, "
                              f"retrying in {wait_time:.1f}s ({retries}/{max_retries})")
                        time.sleep(wait_time)
                    else:
                        break
        
        print(f"File {os.path.basename(fpath)} not found in any of the {len(repos_to_try)} repositories")
        raise last_exception

