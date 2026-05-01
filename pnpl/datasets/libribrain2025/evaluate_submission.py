import os
import csv
import numpy as np
import pandas as pd
from pathlib import Path
import argparse

def load_labels_csv(labels_csv_path):
    """Load the labels CSV file"""
    print(f"Loading labels from: {labels_csv_path}")
    
    try:
        df = pd.read_csv(labels_csv_path)
        labels = df['phoneme_label'].values
        print(f"Loaded {len(labels)} labels")
        return labels
    except Exception as e:
        print(f"Error loading labels: {e}")
        return None

def load_submission_csv(submission_csv_path):
    """Load the submission CSV file and convert probabilities to predictions"""
    print(f"Loading submission from: {submission_csv_path}")
    
    try:
        df = pd.read_csv(submission_csv_path)
        
        # Extract probability columns (phoneme_1 to phoneme_39)
        prob_columns = [f'phoneme_{i+1}' for i in range(39)]
        
        if not all(col in df.columns for col in prob_columns):
            print("Missing phoneme probability columns")
            return None
        
        probabilities = df[prob_columns].values
        
        # Convert probabilities to predictions (argmax)
        predictions = np.argmax(probabilities, axis=1)
        
        print(f"Loaded {len(predictions)} predictions")
        return predictions
        
    except Exception as e:
        print(f"Error loading submission: {e}")
        return None

def calculate_accuracy_metrics(labels, predictions):
    """Calculate basic accuracy metrics"""
    print("\nCalculating accuracy metrics...")
    
    if len(labels) != len(predictions):
        print(f"Length mismatch: labels={len(labels)}, predictions={len(predictions)}")
        return None
    
    # Overall accuracy
    correct = (labels == predictions).sum()
    total = len(labels)
    accuracy = correct / total
    
    print(f"Overall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Correct predictions: {correct}/{total}")
    
    # Per-class accuracy
    unique_labels = np.unique(labels)
    class_accuracies = {}
    
    for label in unique_labels:
        mask = (labels == label)
        if mask.sum() > 0:
            class_correct = (predictions[mask] == label).sum()
            class_total = mask.sum()
            class_acc = class_correct / class_total
            class_accuracies[label] = class_acc
    
    # Summary statistics
    if class_accuracies:
        avg_class_acc = np.mean(list(class_accuracies.values()))
        print(f"Average per-class accuracy: {avg_class_acc:.4f}")
    
    return {
        'overall_accuracy': accuracy,
        'correct': correct,
        'total': total,
        'class_accuracies': class_accuracies
    }

def calculate_confusion_metrics(labels, predictions, num_classes=39):
    """Calculate basic confusion matrix metrics"""
    print("\nCalculating confusion matrix metrics...")
    
    # Create confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(labels, predictions):
        if 0 <= true_label < num_classes and 0 <= pred_label < num_classes:
            confusion_matrix[true_label, pred_label] += 1
    
    # Calculate per-class metrics
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    f1_score = np.zeros(num_classes)
    
    for i in range(num_classes):
        tp = confusion_matrix[i, i]
        fp = confusion_matrix[:, i].sum() - tp
        fn = confusion_matrix[i, :].sum() - tp
        
        precision[i] = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall[i] = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score[i] = 2 * precision[i] * recall[i] / (precision[i] + recall[i]) if (precision[i] + recall[i]) > 0 else 0
    
    # Overall metrics
    macro_precision = precision.mean()
    macro_recall = recall.mean()
    macro_f1 = f1_score.mean()
    
    print(f"Macro Precision: {macro_precision:.4f}")
    print(f"Macro Recall: {macro_recall:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    
    return {
        'confusion_matrix': confusion_matrix,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1
    }

def evaluate_submission(labels_csv_path, submission_csv_path):
    """Main evaluation function"""
    print("Phoneme Submission Evaluator")
    print("=" * 50)
    
    # Check if files exist
    if not Path(labels_csv_path).exists():
        print(f"Labels CSV not found: {labels_csv_path}")
        return None
    
    if not Path(submission_csv_path).exists():
        print(f"Submission CSV not found: {submission_csv_path}")
        return None
    
    # Load data
    labels = load_labels_csv(labels_csv_path)
    if labels is None:
        return None
    
    predictions = load_submission_csv(submission_csv_path)
    if predictions is None:
        return None
    
    # Calculate metrics
    results = {}
    
    # Accuracy metrics
    results['accuracy'] = calculate_accuracy_metrics(labels, predictions)
    if results['accuracy'] is None:
        return None
    
    # Confusion matrix metrics
    results['confusion'] = calculate_confusion_metrics(labels, predictions)
    
    # Summary
    print("\nSummary:")
    print(f"Overall Accuracy: {results['accuracy']['overall_accuracy']:.4f}")
    print(f"Macro F1-Score: {results['confusion']['macro_f1']:.4f}")
    
    return results

def main():
    """Main function with command line interface"""
    parser = argparse.ArgumentParser(description='Evaluate phoneme classification submission')
    parser.add_argument('--labels', type=str, default='../../../miran-holdout-data/phoneme_labels.csv',
                        help='Path to labels CSV file')
    parser.add_argument('--submission', type=str, default='../../../phoneme_submission-1.csv',
                        help='Path to submission CSV file')
    
    args = parser.parse_args()
    
    evaluate_submission(args.labels, args.submission)

if __name__ == "__main__":
    # If run without arguments, use default files
    import sys
    if len(sys.argv) == 1:
        # Default evaluation - adjust paths relative to pnpl package location
        labels_csv = "../../../miran-holdout-data/phoneme_labels.csv"
        submission_csv = "../../../phoneme_submission-1.csv"
        
        if Path(labels_csv).exists() and Path(submission_csv).exists():
            print("Running evaluation with default files...")
            evaluate_submission(labels_csv, submission_csv)
        else:
            print("Default files not found. Please specify paths:")
            print(f"Labels CSV: {labels_csv} {'(exists)' if Path(labels_csv).exists() else '(not found)'}")
            print(f"Submission CSV: {submission_csv} {'(exists)' if Path(submission_csv).exists() else '(not found)'}")
            print("\nUsage: python evaluate_submission.py --labels <labels.csv> --submission <submission.csv>")
    else:
        main() 