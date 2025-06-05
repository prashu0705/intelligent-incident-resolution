import pandas as pd
import numpy as np
import faiss
import ast  # to safely evaluate string embeddings saved as lists

# Load CSV with embeddings
df = pd.read_csv("embeddings/cleaned_incidents_with_embeddings.csv")

# Convert embedding strings back to list (if saved as strings)
def str_to_list(embedding_str):
    try:
        return np.array(ast.literal_eval(embedding_str), dtype='float32')
    except:
        return None

df['embedding'] = df['embedding'].apply(str_to_list)

# Drop rows where embedding is None
df = df.dropna(subset=['embedding']).reset_index(drop=True)

# Stack embeddings into one numpy array
embeddings = np.vstack(df['embedding'].values).astype('float32')

# Get dimension of embeddings
dimension = embeddings.shape[1]

# Build FAISS index (FlatL2 for simplicity, but you can explore other types)
index = faiss.IndexFlatL2(dimension)

# Add vectors to the index
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} vectors of dimension {dimension}")

# Save index for later use
faiss.write_index(index, "incident_embeddings.index")

# Save dataframe as well to map results back to incidents
df.to_pickle("incidents_with_embeddings.pkl")

