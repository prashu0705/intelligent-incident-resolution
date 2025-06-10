import pandas as pd
import numpy as np
import faiss
import ast  # To safely evaluate string representations of lists/arrays
import os

# --- Configuration ---
INPUT_CSV_WITH_EMBEDDINGS_PATH = "embeddings/cleaned_incidents_with_embeddings.csv"
FAISS_INDEX_OUTPUT_PATH = "embeddings/incident_embeddings.index"
DATAFRAME_PKL_OUTPUT_PATH = "embeddings/incidents_with_embeddings.pkl"

# --- Helper Function for Embedding Conversion ---
def str_to_list(embedding_str: str) -> np.ndarray | None:
    """
    Converts a string representation of an embedding (list of floats)
    into a NumPy array of float32.

    The CSV stores embeddings as strings (e.g., "[0.1, 0.2, ...]").
    This function parses that string back into a numerical array.

    Args:
        embedding_str: The string representation of the embedding.

    Returns:
        A NumPy array (float32) of the embedding, or None if conversion fails.
    """
    try:
        # ast.literal_eval is safer than eval for parsing Python literals
        return np.array(ast.literal_eval(embedding_str), dtype='float32')
    except (ValueError, SyntaxError) as e:
        # Log error or handle as appropriate if conversion fails
        print(f"Warning: Could not convert string to list/array: '{embedding_str[:50]}...'. Error: {e}")
        return None

# --- Main Script Logic ---
if __name__ == "__main__":
    print(f"Loading data with embeddings from {INPUT_CSV_WITH_EMBEDDINGS_PATH}...")
    df = pd.read_csv(INPUT_CSV_WITH_EMBEDDINGS_PATH)

    if 'embedding' not in df.columns:
        raise ValueError(f"'embedding' column not found in {INPUT_CSV_WITH_EMBEDDINGS_PATH}.")

    # Convert string representations of embeddings in the 'embedding' column to NumPy arrays
    print("Converting string embeddings to numerical arrays...")
    df['embedding'] = df['embedding'].apply(str_to_list)

    # Drop rows where embedding conversion failed (resulting in None) or where embeddings are missing
    original_row_count = len(df)
    df = df.dropna(subset=['embedding']).reset_index(drop=True)
    if len(df) < original_row_count:
        print(f"Dropped {original_row_count - len(df)} rows due to missing or unparseable embeddings.")

    if df.empty:
        raise ValueError("No valid embeddings found after processing. FAISS index cannot be built.")

    # Stack all embedding vectors into a single NumPy array for FAISS
    print("Stacking embeddings into a NumPy matrix...")
    embeddings_matrix = np.vstack(df['embedding'].values).astype('float32')

    # Get the dimensionality of the embeddings from the matrix shape
    dimension = embeddings_matrix.shape[1]
    print(f"Embeddings matrix created with shape: {embeddings_matrix.shape}")

    # Build the FAISS index
    # Using IndexFlatL2, which performs exact L2 distance search.
    # For larger datasets, more complex index types (e.g., IndexIVFFlat) might be considered for performance.
    print(f"Building FAISS IndexFlatL2 with dimension {dimension}...")
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings_matrix)  # Add all embedding vectors to the index

    print(f"FAISS index built successfully with {index.ntotal} vectors.")

    # Ensure the output directory exists
    output_dir = os.path.dirname(FAISS_INDEX_OUTPUT_PATH) # Assumes both files go to same dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created directory: {output_dir}")

    # Save the built FAISS index to disk
    faiss.write_index(index, FAISS_INDEX_OUTPUT_PATH)
    print(f"FAISS index saved to {FAISS_INDEX_OUTPUT_PATH}")

    # Save the DataFrame (containing metadata associated with embeddings) to a pickle file.
    # This allows mapping search results (indices) back to original incident data.
    df.to_pickle(DATAFRAME_PKL_OUTPUT_PATH)
    print(f"DataFrame with incident data and embeddings saved to {DATAFRAME_PKL_OUTPUT_PATH}")

    print("\nFAISS index building and data pickling complete.")

