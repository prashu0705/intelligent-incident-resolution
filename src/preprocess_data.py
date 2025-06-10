"""
Preprocesses raw incident data from an Excel file.

This script performs the following main operations:
1. Loads data from "data/incident_data.xlsx" (expecting headers in the second row).
2. Handles missing values by dropping rows where 'Summary' or 'Latest Comments' are null.
3. Creates a 'combined_text' field by concatenating 'Summary' and 'Latest Comments'.
4. Selects a specific subset of columns relevant for downstream processing.
5. Saves the cleaned and transformed data to "data/cleaned_incidents.csv".
"""
import pandas as pd

# --- Configuration ---
RAW_DATA_PATH = "data/incident_data.xlsx"
CLEANED_DATA_OUTPUT_PATH = "data/cleaned_incidents.csv"
COLUMNS_TO_SELECT = [
    'Ticket Number', 'Project', 'Category', 'Severity', 'Priority',
    'combined_text', 'Latest Comments'
]

# --- Data Loading ---
# Load raw incident data from the specified Excel file.
# header=1 indicates that the column headers are in the second row of the Excel sheet (0-indexed).
df = pd.read_excel(RAW_DATA_PATH, header=1)
print(f"Loaded {len(df)} rows from {RAW_DATA_PATH}")

# --- Data Cleaning & Transformation ---
# Drop rows if either 'Summary' or 'Latest Comments' fields are missing.
# This is important as these fields are used to create the 'combined_text'.
df.dropna(subset=['Summary', 'Latest Comments'], inplace=True)
print(f"{len(df)} rows remaining after dropping NaNs in 'Summary' or 'Latest Comments'.")

# Combine 'Summary' and 'Latest Comments' into a single text field for NLP tasks.
# Ensure both fields are treated as strings to handle potential numeric or other types.
df['combined_text'] = df['Summary'].astype(str) + " " + df['Latest Comments'].astype(str)

# --- Feature Selection ---
# Select only the relevant columns for the downstream machine learning pipeline.
df_clean = df[COLUMNS_TO_SELECT]
print(f"Selected columns: {', '.join(COLUMNS_TO_SELECT)}")

# --- Data Saving ---
# Save the cleaned and processed DataFrame to a CSV file.
# index=False prevents pandas from writing the DataFrame index as a column.
df_clean.to_csv(CLEANED_DATA_OUTPUT_PATH, index=False)
print(f"Cleaned data saved to {CLEANED_DATA_OUTPUT_PATH}")

# --- Completion ---
print("\nData preprocessing complete. Sample of cleaned data:")
print(df_clean.head())

