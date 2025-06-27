import torch
import numpy as np
import json
import os

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def compute_average_metrics(metrics_list):
    """
    Compute average of metrics across batches
    
    Args:
        metrics_list: List of dictionaries containing metrics
    
    Returns:
        Dictionary with averaged metrics
    """
    if not metrics_list:
        return {}
    
    avg_metrics = {}
    for key in metrics_list[0].keys():
        values = [m[key] for m in metrics_list if key in m]
        if values:
            avg_metrics[key] = sum(values) / len(values)
    
    return avg_metrics

def save_results(results, filepath):
    """Save results to JSON file"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def load_results(filepath):
    """Load results from JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)

def calculate_classification_metrics(predictions, targets, num_classes=200):
    """
    Calculate comprehensive classification metrics
    
    Args:
        predictions: Model predictions [N, num_classes]
        targets: Ground truth labels [N]
        num_classes: Number of classes
    
    Returns:
        Dictionary containing various metrics
    """
    with torch.no_grad():
        # Convert to numpy if needed
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        
        # Get predicted classes
        pred_classes = np.argmax(predictions, axis=1)
        
        # Overall accuracy
        accuracy_score = np.mean(pred_classes == targets) * 100
        
        # Top-5 accuracy
        top5_preds = np.argsort(predictions, axis=1)[:, -5:]
        top5_accuracy = np.mean([targets[i] in top5_preds[i] for i in range(len(targets))]) * 100
        
        # Per-class accuracy
        per_class_acc = []
        for class_id in range(num_classes):
            class_mask = targets == class_id
            if np.sum(class_mask) > 0:
                class_acc = np.mean(pred_classes[class_mask] == targets[class_mask]) * 100
                per_class_acc.append(class_acc)
            else:
                per_class_acc.append(0.0)
        
        metrics = {
            'top1_accuracy': accuracy_score,
            'top5_accuracy': top5_accuracy,
            'per_class_accuracy': per_class_acc,
            'mean_per_class_accuracy': np.mean(per_class_acc),
            'num_samples': len(targets)
        }
        
        return metrics

def print_training_summary(results):
    """
    Print a formatted summary of training results
    
    Args:
        results: Dictionary containing training results
    """
    print("\n" + "="*60)
    print("TRAINING RESULTS SUMMARY")
    print("="*60)
    
    if 'best_val_accuracy' in results:
        print(f"Best Validation Accuracy: {results['best_val_accuracy']:.2f}%")
    
    if 'final_val_accuracy' in results:
        print(f"Final Validation Accuracy: {results['final_val_accuracy']:.2f}%")
    
    if 'training_time' in results:
        print(f"Training Time: {results['training_time']:.2f} seconds")
    
    if 'epochs' in results:
        print(f"Total Epochs: {results['epochs']}")
    
    # Print improvement if baseline is available
    if 'improvement' in results:
        print(f"Improvement over Baseline: +{results['improvement']:.2f}%")
    
    print("="*60)

def compare_experiments(baseline_results, augmented_results):
    """
    Compare baseline and augmented experiment results
    
    Args:
        baseline_results: Results from baseline experiment
        augmented_results: Results from augmented experiment
    
    Returns:
        Dictionary with comparison metrics
    """
    comparison = {}
    
    if 'best_val_accuracy' in baseline_results and 'best_val_accuracy' in augmented_results:
        baseline_acc = baseline_results['best_val_accuracy']
        augmented_acc = augmented_results['best_val_accuracy']
        
        improvement = augmented_acc - baseline_acc
        relative_improvement = (improvement / baseline_acc) * 100 if baseline_acc > 0 else 0
        
        comparison = {
            'baseline_accuracy': baseline_acc,
            'augmented_accuracy': augmented_acc,
            'absolute_improvement': improvement,
            'relative_improvement': relative_improvement
        }
        
        print(f"\nExperiment Comparison:")
        print(f"Baseline: {baseline_acc:.2f}%")
        print(f"Augmented: {augmented_acc:.2f}%")
        print(f"Improvement: +{improvement:.2f}% ({relative_improvement:.1f}% relative)")
    
    return comparison 