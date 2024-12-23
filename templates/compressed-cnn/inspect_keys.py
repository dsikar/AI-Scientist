import numpy as np

# Load the file
results_dict = np.load('run_0/all_results.npy', allow_pickle=True).item()

# Print all keys
print("Keys in all_results.npy:")
for key in results_dict.keys():
    print(f"  {key}")
