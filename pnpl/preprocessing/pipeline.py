"""
Preprocessing Pipeline for MEG data.

Provides a composable pipeline that can be defined explicitly or parsed from a string.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, TYPE_CHECKING
import os

if TYPE_CHECKING:
    import mne


# Step name to class mapping (populated by step imports)
STEP_REGISTRY: Dict[str, type] = {}


def register_step(name: str):
    """Decorator to register a step class with a short name."""
    def decorator(cls):
        STEP_REGISTRY[name] = cls
        cls.step_name = name
        return cls
    return decorator


@dataclass
class Pipeline:
    """
    MEG preprocessing pipeline.
    
    A pipeline is a sequence of preprocessing steps that are applied
    in order to raw MEG data.
    
    Args:
        steps: List of step objects to apply
        
    Example:
        >>> from pnpl.preprocessing import Pipeline, BadChannels, MaxwellFilter
        >>> pipeline = Pipeline([
        ...     BadChannels(),
        ...     MaxwellFilter(),
        ... ])
        >>> 
        >>> # Or from string:
        >>> pipeline = Pipeline.from_string("bads+sss+notch+bp+ds")
    """
    steps: List = field(default_factory=list)
    
    # Context passed between steps
    _context: Dict[str, Any] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        self._context = {}
    
    @classmethod
    def from_string(
        cls,
        spec: str,
        config: Optional[Dict[str, Dict[str, Any]]] = None,
        **kwargs,
    ) -> "Pipeline":
        """
        Parse a pipeline specification string.

        The string format is: step1+step2+step3

        Common shortcuts:
        - bads: BadChannels
        - headpos: HeadPosition
        - sss: MaxwellFilter
        - notch: NotchFilter
        - bp: BandpassFilter
        - ds: Downsample
        - epo: Epoch

        Args:
            spec: Pipeline specification string
            config: Optional dict mapping step names to their configuration parameters.
                    This takes precedence over kwargs.
            **kwargs: Default arguments to pass to steps (deprecated, use config)

        Returns:
            Pipeline instance
        """
        # Import steps to populate registry
        from . import steps as _  # noqa

        config = config or {}
        step_names = spec.split('+')
        step_objects = []

        for name in step_names:
            name = name.strip().lower()
            if not name:
                continue

            if name not in STEP_REGISTRY:
                raise ValueError(f"Unknown step: {name}. Available: {list(STEP_REGISTRY.keys())}")

            step_cls = STEP_REGISTRY[name]
            # Config takes precedence over kwargs
            step_kwargs = kwargs.get(name, {})
            step_kwargs.update(config.get(name, {}))
            step_objects.append(step_cls(**step_kwargs))

        return cls(steps=step_objects)
    
    def to_string(self) -> str:
        """
        Convert pipeline to specification string.
        
        Returns:
            Pipeline specification (e.g., 'bads+headpos+sss+notch+bp+ds')
        """
        names = []
        for step in self.steps:
            if hasattr(step, 'step_name'):
                names.append(step.step_name)
            elif hasattr(step, 'name'):
                names.append(step.name)
            else:
                names.append(type(step).__name__.lower())
        return '+'.join(names)
    
    def run(
        self,
        raw: "mne.io.Raw",
        subject: str,
        session: str,
        task: str,
        run: str,
        bids_root: str,
        verbose: bool = True,
    ) -> "mne.io.Raw":
        """
        Run the pipeline on raw data.
        
        Args:
            raw: MNE Raw object to preprocess
            subject, session, task, run: BIDS identifiers
            bids_root: Path to BIDS root directory
            verbose: Print progress messages
            
        Returns:
            Preprocessed MNE Raw object
        """
        # Initialize context
        self._context = {
            'subject': subject,
            'session': session,
            'task': task,
            'run': run,
            'bids_root': bids_root,
            'head_pos': None,  # Will be set by HeadPosition step
            'bad_channels': {'noisy': [], 'flat': []},
        }
        
        for i, step in enumerate(self.steps, 1):
            if verbose:
                print(f"[{i}/{len(self.steps)}] Running {step.step_name}...")
            
            raw = step.apply(raw, self._context)
        
        return raw
    
    def get_output_filename(
        self,
        subject: str,
        session: str,
        task: str,
        run: str,
        extension: str = "fif",
    ) -> str:
        """
        Generate output filename based on pipeline steps.
        
        Args:
            subject, session, task, run: BIDS identifiers
            extension: File extension ('fif' or 'h5')
            
        Returns:
            Filename like 'sub-0_ses-1_task-X_run-1_proc-bads+sss+..._meg.{ext}'
        """
        proc_str = self.to_string()
        return f"sub-{subject}_ses-{session}_task-{task}_run-{run}_proc-{proc_str}_meg.{extension}"
    
    def get_output_path(
        self,
        bids_root: str,
        subject: str,
        session: str,
        task: str,
        run: str,
        extension: str = "fif",
    ) -> str:
        """
        Generate full output path in derivatives directory.
        
        Args:
            bids_root: BIDS root directory
            subject, session, task, run: BIDS identifiers
            extension: File extension
            
        Returns:
            Full path to output file
        """
        deriv_dir = os.path.join(
            bids_root, "derivatives", "preproc",
            f"sub-{subject}", f"ses-{session}", "meg"
        )
        os.makedirs(deriv_dir, exist_ok=True)
        
        filename = self.get_output_filename(subject, session, task, run, extension)
        return os.path.join(deriv_dir, filename)

