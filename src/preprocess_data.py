import pandas as pd

# Load cleaned data
df = pd.read_excel("data/incident_data.xlsx", header=1)

# Drop rows where 'Summary' or 'Latest Comments' are missing (optional)
df.dropna(subset=['Summary', 'Latest Comments'], inplace=True)

# Combine summary and latest comments for semantic analysis
df['combined_text'] = df['Summary'].astype(str) + " " + df['Latest Comments'].astype(str)

# Optional: select relevant columns for your pipeline
df_clean = df[['Ticket Number', 'Project', 'Category', 'Severity', 'Priority', 'combined_text']]

# Save cleaned CSV for downstream use
df_clean.to_csv("data/cleaned_incidents.csv", index=False)

print(" Data preprocessing complete. Sample:")
print(df_clean.head())

