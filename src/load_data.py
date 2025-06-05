import pandas as pd

# Load Excel file
df = pd.read_excel("data/incident_data.xlsx", header=1)

# Display basic info
print("Columns:", df.columns.tolist())
print("\nSample rows:")
print(df.head())

