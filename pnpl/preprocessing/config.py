"""
Preprocessing Configuration Resolution Module.

Implements a cascading configuration system for preprocessing steps with three priority levels:
1. Step defaults (lowest) - dataclass field defaults in each step
2. JSON config file (medium) - {data_path}/preprocessing_config.json
3. Dataset config (highest) - preprocessing_config parameter on dataset
"""

import json
import os
from dataclasses import dataclass, fields, field
from typing import Dict, Any, List, Optional


@dataclass
class ResolvedConfig:
    """Resolved preprocessing configuration with source tracking."""

    config: Dict[str, Dict[str, Any]]  # step_name -> {param: value}
    sources: Dict[str, Dict[str, str]]  # step_name -> {param: source}

    def get_log_message(self) -> str:
        """
        Generate a log message summarizing the configuration sources.

        Examples:
        - All defaults: "Preprocessing config: using defaults"
        - Single source: "Preprocessing config: bp, ds (JSON config), rest defaults"
        - Mixed: "Preprocessing config: bp.l_freq (dataset), ds (JSON config), rest defaults"
        - All custom: "Preprocessing config: bp (dataset), ds, notch (JSON config)"
        """
        # Collect params by source
        dataset_params: List[str] = []
        json_params: List[str] = []
        has_defaults = False

        for step_name, params in self.sources.items():
            step_dataset_params = []
            step_json_params = []
            step_has_defaults = False

            for param, source in params.items():
                if source == "dataset":
                    step_dataset_params.append(param)
                elif source == "JSON config":
                    step_json_params.append(param)
                else:
                    step_has_defaults = True

            # Determine how to represent this step
            total_params = len(params)

            if len(step_dataset_params) == total_params:
                # All params from dataset
                dataset_params.append(step_name)
            elif len(step_json_params) == total_params:
                # All params from JSON
                json_params.append(step_name)
            elif step_has_defaults and not step_dataset_params and not step_json_params:
                # All defaults
                has_defaults = True
            else:
                # Mixed sources - need to specify individual params
                for param in step_dataset_params:
                    dataset_params.append(f"{step_name}.{param}")
                for param in step_json_params:
                    json_params.append(f"{step_name}.{param}")
                if step_has_defaults:
                    has_defaults = True

        # Build message
        parts = []

        if dataset_params:
            parts.append(f"{', '.join(dataset_params)} (dataset)")

        if json_params:
            parts.append(f"{', '.join(json_params)} (JSON config)")

        if has_defaults:
            parts.append("rest defaults")

        if not parts:
            return "Preprocessing config: using defaults"

        if parts == ["rest defaults"]:
            return "Preprocessing config: using defaults"

        return f"Preprocessing config: {', '.join(parts)}"


def get_step_defaults(step_name: str) -> Dict[str, Any]:
    """
    Extract default values from a step's dataclass fields.

    Args:
        step_name: The registered name of the step (e.g., 'bp', 'ds', 'notch')

    Returns:
        Dictionary mapping parameter names to their default values
    """
    from .pipeline import STEP_REGISTRY
    from . import steps as _  # noqa: Ensure steps are registered

    if step_name not in STEP_REGISTRY:
        return {}

    step_cls = STEP_REGISTRY[step_name]
    defaults = {}

    for f in fields(step_cls):
        # Skip step_name field as it's not a configurable parameter
        if f.name == "step_name":
            continue

        # Get default value
        if f.default is not dataclass_field_missing():
            defaults[f.name] = f.default
        elif f.default_factory is not dataclass_field_missing():
            defaults[f.name] = f.default_factory()

    return defaults


def dataclass_field_missing():
    """Helper to check for missing dataclass field defaults."""
    from dataclasses import MISSING
    return MISSING


def load_json_config(data_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load preprocessing_config.json if it exists.

    Args:
        data_path: Path to the data directory

    Returns:
        Dictionary mapping step names to their configuration parameters,
        or empty dict if file doesn't exist
    """
    config_path = os.path.join(data_path, "preprocessing_config.json")

    if not os.path.exists(config_path):
        return {}

    try:
        with open(config_path, "r") as f:
            config = json.load(f)

        # Validate structure
        if not isinstance(config, dict):
            return {}

        return config
    except (json.JSONDecodeError, IOError):
        return {}


def resolve_preprocessing_config(
    step_names: List[str],
    json_config: Optional[Dict[str, Dict[str, Any]]] = None,
    dataset_config: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ResolvedConfig:
    """
    Merge configs with precedence: defaults < JSON < dataset.

    Args:
        step_names: List of step names in the pipeline
        json_config: Configuration from preprocessing_config.json
        dataset_config: Configuration passed to dataset constructor

    Returns:
        ResolvedConfig with merged configuration and source tracking
    """
    json_config = json_config or {}
    dataset_config = dataset_config or {}

    resolved: Dict[str, Dict[str, Any]] = {}
    sources: Dict[str, Dict[str, str]] = {}

    for step in step_names:
        step = step.strip().lower()
        if not step:
            continue

        step_defaults = get_step_defaults(step)
        resolved[step] = {}
        sources[step] = {}

        for param, default_value in step_defaults.items():
            # Start with default
            value = default_value
            source = "defaults"

            # Override with JSON config
            if step in json_config and param in json_config[step]:
                value = json_config[step][param]
                source = "JSON config"

            # Override with dataset config
            if step in dataset_config and param in dataset_config[step]:
                value = dataset_config[step][param]
                source = "dataset"

            resolved[step][param] = value
            sources[step][param] = source

    return ResolvedConfig(config=resolved, sources=sources)
