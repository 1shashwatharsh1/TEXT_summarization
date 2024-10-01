import pandas as pd

# Load the dataset
data = pd.read_csv('data/impression_data.csv')  # Adjust this to your actual dataset name

# Print the column names
print("Column names in the dataset:")
print(data.columns.tolist())
