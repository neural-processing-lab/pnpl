"""
Data Resolution Module.

Implements the logic for resolving data files:
1. Check if preprocessed H5 exists locally
2. Check if available on HuggingFace for download
3. If raw BIDS data available, run preprocessing pipeline
4. Serialize to H5 for future use
"""

import os
from typing import Optional, Tuple, Dict, Any, TYPE_CHECKING

if TYPE_CHECKING:
    import mne
    from ..preprocessing import Pipeline


class DataResolver:
    """
    Resolves data files for a dataset.
    
    Handles the flow:
    Local H5 -> HuggingFace Download -> Run Preprocessing -> Serialize
    """
    
    def __init__(
        self,
        data_path: str,
        preprocessing: str,
        download: bool = True,
        huggingface_repo: str = "",
        fallback_repos: list = None,
        preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        """
        Initialize the data resolver.

        Args:
            data_path: Local data directory
            preprocessing: Preprocessing string (e.g., 'bads+headpos+sss+notch+bp+ds')
            download: Whether to download from HuggingFace
            huggingface_repo: Primary HuggingFace repository
            fallback_repos: Fallback HuggingFace repositories
            preprocessing_config: Optional dict of step configs that override defaults
        """
        self.data_path = data_path
        self.preprocessing = preprocessing
        self.download = download
        self.huggingface_repo = huggingface_repo
        self.fallback_repos = fallback_repos or []
        self.preprocessing_config = preprocessing_config or {}
    
    def resolve_h5(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> Tuple[str, str]:
        """
        Resolve H5 file for a run.
        
        Returns:
            Tuple of (h5_path, resolution_method)
            resolution_method is one of: 'local', 'download', 'preprocess'
        """
        h5_path = self._get_h5_path(subject, session, task, run)
        
        # 1. Check local
        if os.path.exists(h5_path):
            return h5_path, 'local'
        
        # 2. Try HuggingFace download
        if self.download:
            try:
                downloaded = self._try_download(h5_path, subject, session, task, run)
                if downloaded:
                    return h5_path, 'download'
            except Exception:
                pass
        
        # 3. Try preprocessing from raw BIDS
        raw_path = self._get_raw_bids_path(subject, session, task, run)
        if os.path.exists(raw_path):
            self._run_preprocessing(subject, session, task, run, h5_path)
            return h5_path, 'preprocess'
        
        # 4. No data available
        raise FileNotFoundError(
            f"Could not find or create data for sub-{subject}/ses-{session}/task-{task}/run-{run}. "
            f"Checked: local H5, HuggingFace download, raw BIDS preprocessing."
        )
    
    def _get_h5_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> str:
        """Get path to H5 file."""
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}"
        if self.preprocessing:
            fname += f"_proc-{self.preprocessing}"
        fname += "_meg.h5"
        return os.path.join(self.data_path, task, "derivatives", "serialised", fname)
    
    def _get_raw_bids_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> str:
        """Get path to raw BIDS MEG file."""
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_meg.fif"
        return os.path.join(
            self.data_path,
            f"sub-{subject}",
            f"ses-{session}",
            "meg",
            fname,
        )
    
    def _try_download(
        self,
        h5_path: str,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> bool:
        """
        Try to download H5 file from HuggingFace.
        
        Returns:
            True if download successful, False otherwise
        """
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import RepositoryNotFoundError, EntryNotFoundError
        
        rel_path = os.path.relpath(h5_path, self.data_path)
        rel_path = rel_path.replace(os.path.sep, '/')
        
        repos = [self.huggingface_repo] + self.fallback_repos
        
        for repo in repos:
            if not repo:
                continue
            try:
                hf_hub_download(
                    repo_id=repo,
                    repo_type="dataset",
                    filename=rel_path,
                    local_dir=self.data_path,
                )
                return True
            except (RepositoryNotFoundError, EntryNotFoundError):
                continue
            except Exception:
                continue
        
        return False
    
    def _run_preprocessing(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        output_h5_path: str,
    ) -> None:
        """
        Run preprocessing pipeline on raw BIDS data.
        """
        import mne
        from ..preprocessing import Pipeline
        from ..preprocessing.serialization import fif_to_h5
        from ..preprocessing.config import (
            resolve_preprocessing_config,
            load_json_config,
        )

        print(f"Preprocessing: sub-{subject}/ses-{session}/task-{task}/run-{run}")

        # Resolve preprocessing configuration with cascading precedence
        step_names = self.preprocessing.split('+')
        json_config = load_json_config(self.data_path)
        resolved = resolve_preprocessing_config(
            step_names=step_names,
            json_config=json_config,
            dataset_config=self.preprocessing_config,
        )

        # Log the configuration sources
        print(resolved.get_log_message())

        # Load raw data
        raw_path = self._get_raw_bids_path(subject, session, task, run)
        raw = mne.io.read_raw_fif(raw_path, preload=True, verbose=False)

        # Create and run pipeline with resolved config
        pipeline = Pipeline.from_string(self.preprocessing, config=resolved.config)
        raw = pipeline.run(
            raw,
            subject=subject,
            session=session,
            task=task,
            run=run,
            bids_root=self.data_path,
            verbose=True,
        )

        # Save as FIF first
        fif_path = pipeline.get_output_path(
            self.data_path, subject, session, task, run, extension="fif"
        )
        raw.save(fif_path, overwrite=True, verbose=False)

        # Serialize to H5
        os.makedirs(os.path.dirname(output_h5_path), exist_ok=True)
        fif_to_h5(raw, output_h5_path)

        print(f"Saved: {output_h5_path}")


def resolve_data(
    data_path: str,
    preprocessing: str,
    subject: str,
    session: str,
    task: str,
    run: str,
    download: bool = True,
    huggingface_repo: str = "",
    fallback_repos: list = None,
    preprocessing_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Tuple[str, str]:
    """
    Convenience function to resolve data for a single run.

    Returns:
        Tuple of (h5_path, resolution_method)
    """
    resolver = DataResolver(
        data_path=data_path,
        preprocessing=preprocessing,
        download=download,
        huggingface_repo=huggingface_repo,
        fallback_repos=fallback_repos,
        preprocessing_config=preprocessing_config,
    )
    return resolver.resolve_h5(subject, session, task, run)

