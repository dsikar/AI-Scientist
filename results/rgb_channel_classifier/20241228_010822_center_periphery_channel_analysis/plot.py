import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Style settings
plt.style.use('seaborn-v0_8')
colors = ['#FF7F50', '#4169E1', '#32CD32']  # coral, royal blue, lime green

# Dictionary mapping run directories to their labels
labels = {
    'run_0': 'Baseline (No Mask)',
    'run_3': 'B Channel',
    'run_4': 'R+G Channels'
}

def load_results(run_dir):
    """Load results from a run directory."""
    with open(os.path.join(run_dir, 'final_info.json'), 'r') as f:
        return json.load(f)

def create_accuracy_comparison():
    """Create bar plot comparing accuracies across runs."""
    accuracies = []
    names = []
    
    for run_dir, label in labels.items():
        if os.path.exists(run_dir):
            results = load_results(run_dir)
            acc = results['cifar10']['means']['test_acc_mean']
            accuracies.append(acc)
            names.append(label)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, accuracies)
    plt.title('Test Accuracy Comparison Across Experiments')
    plt.ylabel('Accuracy (%)')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()

def create_training_time_comparison():
    """Create bar plot comparing training times across runs."""
    times = []
    names = []
    
    for run_dir, label in labels.items():
        if os.path.exists(run_dir):
            results = load_results(run_dir)
            time = results['cifar10']['means']['total_train_time_mean'] / 60  # Convert to minutes
            times.append(time)
            names.append(label)
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(names, times)
    plt.title('Training Time Comparison Across Experiments')
    plt.ylabel('Training Time (minutes)')
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_time_comparison.png')
    plt.close()

def create_performance_drop_visualization():
    """Create visualization of performance drop from baseline."""
    baseline = None
    drops = []
    names = []
    
    for run_dir, label in labels.items():
        if os.path.exists(run_dir):
            results = load_results(run_dir)
            acc = results['cifar10']['means']['test_acc_mean']
            
            if label == 'Baseline (No Mask)':
                baseline = acc
            else:
                drops.append(baseline - acc)
                names.append(label)
    
    if baseline is not None:
        plt.figure(figsize=(10, 6))
        bars = plt.bar(names, drops)
        plt.title('Performance Drop from Baseline')
        plt.ylabel('Accuracy Drop (%)')
        
        # Add value labels on top of bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}%',
                    ha='center', va='bottom')
        
        plt.tight_layout()
        plt.savefig('performance_drop.png')
        plt.close()

def main():
    """Generate all plots."""
    create_accuracy_comparison()
    create_training_time_comparison()
    create_performance_drop_visualization()
    
    print("Plots have been generated:")
    print("1. accuracy_comparison.png - Compare test accuracies across experiments")
    print("2. training_time_comparison.png - Compare training times across experiments")
    print("3. performance_drop.png - Visualize performance drop from baseline")

if __name__ == "__main__":
    main()
