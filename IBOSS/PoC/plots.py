import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

df_orig = pd.read_csv('https://raw.githubusercontent.com/rathi-mohit/SUBLIME_Mohit/refs/heads/main/IBOSS/PoC/Gen/input1k.csv')
df_sel = pd.read_csv('https://raw.githubusercontent.com/rathi-mohit/SUBLIME_Mohit/refs/heads/main/IBOSS/PoC/Gen/output1k.csv')

def get_matched_data(original, selected, tolerance=1e-4):
    tree = cKDTree(original[['X1', 'X2']].values)
    distances, idxs = tree.query(selected[['X1', 'X2']].values)
    
    valid_mask = distances < tolerance
    matched_indices = idxs[valid_mask]
    
    if len(matched_indices) < len(selected):
        print(f"Warning: {len(selected) - len(matched_indices)} points dropped due to tolerance.")
        
    return original.iloc[matched_indices]

matched_subset = get_matched_data(df_orig, df_sel)

plt.figure(figsize=(10, 8))

plt.scatter(df_orig['X1'], df_orig['X2'], c='gray', alpha=0.5, s=20, label='Original Data')

plt.scatter(matched_subset['X1'], matched_subset['X2'], 
            facecolors='none', 
            edgecolors='red', 
            marker='s', 
            s=150, 
            linewidth=2, 
            alpha=0.7, 
            label='Selected')

plt.title('X1 vs X2 Distribution')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

plt.savefig('highlighted_plot.png')
plt.show()

print(matched_subset.tail())