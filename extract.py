import pandas as pd
from pathlib import Path

# Path to input CSV
input_path = Path.cwd() / "data" / "cleaned_incidents.csv"

# Path to output file in Downloads
output_path = Path.home() / "Downloads" / "diagnosis_knowledge_base.txt"

# Function to format each incident chunk
def create_chunk(row):
    return f"""Ticket Number: {row['Ticket Number']}
Project: {row['Project']}
Category: {row['Category']}
Severity: {row['Severity']}
Priority: {row['Priority']}
Incident Description:
{row['combined_text']}
{'-'*60}

"""

# Read CSV
df = pd.read_csv(input_path)

# Generate chunks
chunks = [create_chunk(row) for _, row in df.iterrows()]

# Write chunks to output file
with open(output_path, "w", encoding="utf-8") as f:
    f.writelines(chunks)

print(f"✅ Knowledge base saved to {output_path}")

