import json
import os
import matplotlib.pyplot as plt
import numpy as np

# Dictionary mapping run directories to their display labels
labels = {
    'run_0': 'Red Channel',
    'run_1': 'Green Channel',
    'run_2': 'Blue Channel',
    'run_3': 'Red+Green',
    'run_4': 'All Channels'
}

def load_results(run_dir):
    """Load results from a run directory."""
    with open(os.path.join(run_dir, 'final_info.json'), 'r') as f:
        return json.load(f)

def create_accuracy_plot(results_dict):
    """Create bar plot comparing accuracies."""
    plt.figure(figsize=(10, 6))
    
    accuracies = []
    names = []
    
    for run_dir, label in labels.items():
        if os.path.exists(run_dir):
            results = load_results(run_dir)
            acc = results['cifar10']['means']['test_acc_mean']
            accuracies.append(acc)
            names.append(label)
    
    bars = plt.bar(range(len(accuracies)), accuracies)
    plt.ylabel('Test Accuracy (%)')
    plt.title('Classification Accuracy by Channel Configuration')
    plt.xticks(range(len(accuracies)), names, rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('accuracy_comparison.png')
    plt.close()

def create_time_plot(results_dict):
    """Create bar plot comparing training times."""
    plt.figure(figsize=(10, 6))
    
    times = []
    names = []
    
    for run_dir, label in labels.items():
        if os.path.exists(run_dir):
            results = load_results(run_dir)
            time = results['cifar10']['means']['total_train_time_mean'] / 60  # Convert to minutes
            times.append(time)
            names.append(label)
    
    bars = plt.bar(range(len(times)), times)
    plt.ylabel('Training Time (minutes)')
    plt.title('Training Time by Channel Configuration')
    plt.xticks(range(len(times)), names, rotation=45)
    
    # Add value labels on top of bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}m',
                ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig('training_time_comparison.png')
    plt.close()

def main():
    # Create results dictionary
    results_dict = {}
    for run_dir in labels.keys():
        if os.path.exists(run_dir):
            results_dict[run_dir] = load_results(run_dir)
    
    # Generate plots
    create_accuracy_plot(results_dict)
    create_time_plot(results_dict)
    
    print("Plots have been saved as 'accuracy_comparison.png' and 'training_time_comparison.png'")

if __name__ == '__main__':
    main()
