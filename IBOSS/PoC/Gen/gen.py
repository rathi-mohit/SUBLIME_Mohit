# Code for generation: Files saved in GitHub
 
import pandas as pd
import numpy as np

N = 10000         # Define the number of datapoints here
output_file = "input.csv"

mu_x1, sigma_x1 = 0, 1     # X1: Mean = 0, Std Dev = 1
mu_x2, sigma_x2 = 0, 10    # X2: Mean = 50, Std Dev = 100

np.random.seed(42) # Seed (for reproducibility)

data = {
    'X1': np.random.normal(mu_x1, sigma_x1, N),
    'X2': np.random.normal(mu_x2, sigma_x2, N)
}

df = pd.DataFrame(data)

df.to_csv(output_file, index=False)

print(f"Successfully generated {N} datapoints and saved to '{output_file}'")
print(df.head())