"""
BIDSMixin - Provides loading for raw BIDS-formatted MEG data.

This mixin handles reading raw MEG data from BIDS directory structure:
- sub-{subject}/ses-{session}/meg/sub-{subject}_ses-{session}_task-{task}_run-{run}_meg.fif

Used when preprocessing needs to be run locally on raw data.
"""

import os
from pathlib import Path
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    import mne


class BIDSMixin:
    """
    Mixin for loading raw BIDS MEG data.
    
    Classes using this mixin should have:
    - data_path: str - Base data directory (BIDS root)
    """
    
    def get_bids_raw_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> str:
        """
        Construct path to raw BIDS MEG file.
        
        Args:
            subject, session, task, run: BIDS identifiers
            
        Returns:
            Full path to raw .fif file
        """
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_meg.fif"
        return os.path.join(
            self.data_path,
            f"sub-{subject}",
            f"ses-{session}",
            "meg",
            fname,
        )
    
    def get_derivatives_path(
        self,
        subject: str,
        session: str,
        derivative_type: str = "preproc",
    ) -> str:
        """
        Construct path to derivatives directory.
        
        Args:
            subject, session: BIDS identifiers
            derivative_type: Type of derivative ('preproc', 'neo', etc.)
            
        Returns:
            Path to derivatives directory
        """
        return os.path.join(
            self.data_path,
            "derivatives",
            derivative_type,
            f"sub-{subject}",
            f"ses-{session}",
            "meg",
        )
    
    def get_calibration_files(self) -> dict:
        """
        Get paths to Maxwell filter calibration files.
        
        Returns:
            Dict with 'calibration' and 'cross_talk' paths
        """
        neo_dir = os.path.join(self.data_path, "derivatives", "neo")
        return {
            "calibration": os.path.join(neo_dir, "sss_cal.dat"),
            "cross_talk": os.path.join(neo_dir, "ct_sparse.fif"),
        }
    
    def get_headpos_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> str:
        """
        Construct path to cached head position file.
        
        Args:
            subject, session, task, run: BIDS identifiers
            
        Returns:
            Path to head position CSV file
        """
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}.csv"
        return os.path.join(
            self.data_path,
            "derivatives",
            "preproc",
            "headpos",
            fname,
        )
    
    def load_raw_bids(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        preload: bool = True,
    ) -> "mne.io.Raw":
        """
        Load raw MEG data from BIDS structure.
        
        Args:
            subject, session, task, run: BIDS identifiers
            preload: Whether to load data into memory
            
        Returns:
            MNE Raw object
        """
        import mne
        
        raw_path = self.get_bids_raw_path(subject, session, task, run)

        # If the raw file isn't present locally, try to fetch it using the dataset's
        # downloader when available (e.g. HFDownloadMixin/OhanaDownloadMixin).
        if not os.path.exists(raw_path):
            if hasattr(self, "ensure_file") and getattr(self, "download", True):
                try:
                    raw_path = self.ensure_file(raw_path)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Raw BIDS file not found and download failed: {raw_path}"
                    ) from e
            else:
                raise FileNotFoundError(f"Raw BIDS file not found: {raw_path}")
        
        return mne.io.read_raw_fif(raw_path, preload=preload, verbose=False)
    
    def load_head_positions(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ):
        """
        Load cached head positions from CSV file.
        
        Args:
            subject, session, task, run: BIDS identifiers
            
        Returns:
            MNE head position array
        """
        import mne
        
        headpos_path = self.get_headpos_path(subject, session, task, run)
        
        if not os.path.exists(headpos_path):
            raise FileNotFoundError(f"Head position file not found: {headpos_path}")
        
        return mne.chpi.read_head_pos(headpos_path)
    
    def raw_bids_exists(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
    ) -> bool:
        """Check if raw BIDS data exists for given identifiers."""
        return os.path.exists(self.get_bids_raw_path(subject, session, task, run))
    
    def get_preprocessed_path(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        preprocessing: str,
        extension: str = "fif",
    ) -> str:
        """
        Construct path to preprocessed file in derivatives.
        
        Args:
            subject, session, task, run: BIDS identifiers
            preprocessing: Preprocessing string (e.g., 'bads+headpos+sss+notch+bp+ds')
            extension: File extension ('fif' or 'h5')
            
        Returns:
            Path to preprocessed file
        """
        deriv_dir = self.get_derivatives_path(subject, session, "preproc")
        fname = f"sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{preprocessing}_meg.{extension}"
        return os.path.join(deriv_dir, fname)

    def load_preprocessed_bids(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        preprocessing: str,
        preload: bool = True,
    ) -> "mne.io.Raw":
        """
        Load a preprocessed FIF file from the derivatives directory.

        Args:
            subject, session, task, run: BIDS identifiers
            preprocessing: Preprocessing string embedded in the filename
            preload: Whether to load data into memory

        Returns:
            MNE Raw object
        """
        import mne

        fif_path = self.get_preprocessed_path(
            subject,
            session,
            task,
            run,
            preprocessing=preprocessing,
            extension="fif",
        )
        if not os.path.exists(fif_path):
            if hasattr(self, "ensure_file") and getattr(self, "download", True):
                try:
                    fif_path = self.ensure_file(fif_path)
                except Exception as e:
                    raise FileNotFoundError(
                        f"Preprocessed FIF file not found and download failed: {fif_path}"
                    ) from e
            else:
                raise FileNotFoundError(f"Preprocessed FIF file not found: {fif_path}")

        return mne.io.read_raw_fif(fif_path, preload=preload, verbose=False)
