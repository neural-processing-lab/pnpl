"""
Example: Loading LibriBrain datasets.

This example shows both the task-based dataset entry point and wrapper datasets.
"""

# =============================================================================
# TASK-BASED DATASET ENTRY POINT
# =============================================================================

from pnpl.datasets import LibriBrain
from pnpl.tasks import SpeechDetection, PhonemeClassification


def task_based_example():
    """Example using the task-based dataset entry point."""
    print("=== TASK-BASED DATASET ENTRY POINT ===\n")
    
    # Speech detection task
    speech_task = SpeechDetection(tmin=0.0, tmax=0.8)
    
    speech_data = LibriBrain(
        data_path="./data/",
        task=speech_task,
        include_run_keys=[("0", "1", "Sherlock1", "1")],
    )
    
    sample, label = speech_data[0]
    print("Speech Detection:")
    print(f"  Data shape: {sample.shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Task info: {speech_task.label_info}")
    print()
    
    # Phoneme classification task
    phoneme_task = PhonemeClassification(tmin=0.0, tmax=0.8)
    
    phoneme_data = LibriBrain(
        data_path="./data/",
        task=phoneme_task,
        include_run_keys=[
            ("0", "1", "Sherlock1", "1"),
            ("0", "1", "Sherlock2", "1"),
            ("0", "1", "Sherlock3", "1"),
        ],
    )
    
    sample, label = phoneme_data[0]
    print("Phoneme Classification:")
    print(f"  Data shape: {sample.shape}")
    print(f"  Label: {label}")
    print(f"  Num classes: {phoneme_task.label_info['n_classes']}")
    print()


# =============================================================================
# WRAPPER DATASETS
# =============================================================================

from pnpl.datasets import LibriBrainSpeech, LibriBrainPhoneme


def wrapper_example():
    """Example using dataset-specific wrapper classes."""
    print("=== WRAPPER DATASETS ===\n")
    
    # Speech detection wrapper
    speech_example_data = LibriBrainSpeech(
        data_path="./data/",
        include_run_keys=[("0", "1", "Sherlock1", "1")],
        tmin=0.0,
        tmax=0.8,
    )
    sample_data, label = speech_example_data[0]

    print("Speech/Non-Speech:")
    print(f"  Sample data shape: {sample_data.shape}")
    print(f"  Label shape: {label.shape}")
    print()

    # Phoneme classification wrapper
    phoneme_example_data = LibriBrainPhoneme(
        data_path="./data/",
        include_run_keys=[
            ("0", "1", "Sherlock1", "1"),
            ("0", "1", "Sherlock2", "1"),
            ("0", "1", "Sherlock3", "1"),
        ],
        tmin=0.0,
        tmax=0.8,
    )
    sample_data, label = phoneme_example_data[0]

    print("Phoneme Classification:")
    print(f"  Sample data shape: {sample_data.shape}")
    print(f"  Label: {label}")
    print()


if __name__ == '__main__':
    task_based_example()
    wrapper_example()
