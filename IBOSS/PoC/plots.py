import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# --- LOAD YOUR DATA HERE ---
df_original = pd.read_csv('https://raw.githubusercontent.com/rathi-mohit/SUBLIME_Mohit/refs/heads/main/IBOSS/PoC/Gen/input.csv')
df_selected = pd.read_csv('https://raw.githubusercontent.com/rathi-mohit/SUBLIME_Mohit/refs/heads/main/IBOSS/PoC/Gen/output.csv')

def get_matched_data(original, selected, tolerance=1e-4):
    """
    Finds the corresponding points in 'original' for each point in 'selected'
    using nearest neighbor search to handle precision issues.
    """
    # Build a KDTree for efficient spatial searching
    tree = cKDTree(original[['X1', 'X2']].values)
    
    # Query the tree to find the nearest original point for every selected point
    # distances: distance to the nearest neighbor
    # idxs: the index of that neighbor in the original dataframe
    distances, idxs = tree.query(selected[['X1', 'X2']].values)
    
    # Optional: Filter out matches that are too far away (sanity check)
    valid_mask = distances < tolerance
    matched_indices = idxs[valid_mask]
    
    if len(matched_indices) < len(selected):
        print(f"Warning: {len(selected) - len(matched_indices)} points could not be matched within tolerance.")
        
    return original.iloc[matched_indices]

# Get the "clean" coordinates from the original set that match the "noisy" selected set
matched_subset = get_matched_data(df_original, df_selected)

# --- PLOTTING ---
plt.figure(figsize=(10, 8))

# 1. Scatter the Original Data (Background)
plt.scatter(df_original['X1'], df_original['X2'], 
            c='gray', alpha=0.5, s=20, label='Original Data')

# 2. Highlight the Selected Data
# specific trick: we plot the MATCHED coordinates, not the raw selected ones.
# This ensures the box is perfectly centered on the background dot.
plt.scatter(matched_subset['X1'], matched_subset['X2'], 
            facecolors='none',    # No fill (transparent center)
            edgecolors='red',     # Red border
            marker='s',           # Square shape
            s=150,                # Size of the box
            linewidth=2, 
            alpha=0.7, 
            label='Selected (Highlighted)')

plt.title('X1 vs X2: Robust Highlighting of Selected Data')
plt.xlabel('X1')
plt.ylabel('X2')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)

# Save or show
plt.savefig('highlighted_plot.png')
plt.show()
print(matched_subset.tail)