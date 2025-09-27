import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_data():
    """Reads CSVs and generates the requested plots."""
    try:
        # Load the data from the user-provided CSV files
        df_full = pd.read_csv('input.csv')        # Change filename to your filename
        df_selected = pd.read_csv('output.csv')   # Change filename to your filename
    except FileNotFoundError:
        print("Error: One or both CSV files not found. Please ensure they are in the same directory.")
        return
    except Exception as e:
        print(f"An error occurred while reading the CSV files: {e}")
        return

    # Clean column names by stripping whitespace
    df_full.columns = df_full.columns.str.strip()
    df_selected.columns = df_selected.columns.str.strip()
    
    # Check if the required columns exist after cleaning
    if 'X1' not in df_full.columns or 'X2' not in df_full.columns:
        print("Error: Required columns 'X1' or 'X2' not found in normal_data.csv after cleaning.")
        return
    if 'X1' not in df_selected.columns or 'X2' not in df_selected.columns:
        print("Error: Required columns 'X1' or 'X2' not found in output.csv after cleaning.")
        return

    # Truncate the data to 4 decimal places for accurate comparison
    df_full['X1'] = np.trunc(df_full['X1'] * 10000) / 10000
    df_full['X2'] = np.trunc(df_full['X2'] * 10000) / 10000
    df_selected['X1'] = np.trunc(df_selected['X1'] * 10000) / 10000
    df_selected['X2'] = np.trunc(df_selected['X2'] * 10000) / 10000

    # Create a figure and a set of subplots for the new plots (2 rows, 1 column)
    fig, axes = plt.subplots(2, 1, figsize=(8, 12))
    fig.suptitle('X1 vs X2 Plots', fontsize=16)

    # Plot 1: X1 vs X2 for all data
    axes[0].scatter(df_full['X1'], df_full['X2'], c='blue', alpha=0.6)
    axes[0].set_title('Plot of X1 vs X2 (All Data)')
    axes[0].set_xlabel('X1')
    axes[0].set_ylabel('X2')
    axes[0].grid(False) # Grid removed

    # Plot 2: X1 vs X2 for selected data
    axes[1].scatter(df_selected['X1'], df_selected['X2'], c='orange', s=100, edgecolors='black', alpha=0.8)
    axes[1].set_title('Plot of X1 vs X2 (Selected Data)')
    axes[1].set_xlabel('X1')
    axes[1].set_ylabel('X2')
    axes[1].grid(False) # Grid removed

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()

# Run the plotting function
if __name__ == "__main__":
    plot_data()
