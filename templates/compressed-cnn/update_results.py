import numpy as np

# Load the existing file
results_dict = np.load('run_0/all_results.npy', allow_pickle=True).item()

# Update each seed's config with dataset field
for key in results_dict.keys():
    if 'final_info' in key:
        results_dict[key]['config']['dataset'] = 'mnist'

# Save the updated file
np.save('run_0/all_results.npy', results_dict)
print("Updated all_results.npy with dataset field")
