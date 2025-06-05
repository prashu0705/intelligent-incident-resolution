import os
import pandas as pd
from openai import AzureOpenAI
from dotenv import load_dotenv
from tqdm import tqdm
import re

# Load environment variables
load_dotenv()

# Initialize OpenAI client for Azure
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)

embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Function to clean text
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9 .,!?]', '', text)
    return text.strip()

# Load your CSV file
df = pd.read_csv("data/cleaned_incidents.csv")

# Clean the text column
df['cleaned_text'] = df['combined_text'].apply(clean_text)

# Generate embeddings
embeddings = []
print("Generating embeddings for each incident...")

for text in tqdm(df['cleaned_text'], total=len(df)):
    if text == "":
        embeddings.append(None)
        continue
    try:
        response = client.embeddings.create(
            model=embedding_deployment,
            input=text
        )
        emb = response.data[0].embedding
        embeddings.append(emb)
    except Exception as e:
        print(f"Error generating embedding: {e}")
        embeddings.append(None)

# Save embeddings
df['embedding'] = embeddings
df.to_csv("cleaned_incidents_with_embeddings.csv", index=False)
print("Embedding generation complete! Saved to cleaned_incidents_with_embeddings.csv")

