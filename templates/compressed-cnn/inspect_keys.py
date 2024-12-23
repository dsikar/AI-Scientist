import numpy as np

# Load the file
results_dict = np.load('run_0/all_results.npy', allow_pickle=True).item()

# Print all keys and their nested structure
print("Keys and structure in all_results.npy:")
for key in results_dict.keys():
    print(f"\n{key}:")
    seed_data = results_dict[key]
    if isinstance(seed_data, dict):
        for subkey, value in seed_data.items():
            print(f"  {subkey}: {type(value)}")
            if isinstance(value, (list, dict)):
                print(f"    {value}")
