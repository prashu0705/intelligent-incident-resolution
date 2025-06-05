import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from openai import AzureOpenAI
import os

app = FastAPI()

# Load environment variables and initialize OpenAI Azure client
from dotenv import load_dotenv
load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Load FAISS index and DataFrame on startup
index = faiss.read_index("embeddings/incident_embeddings.index")
df = pd.read_pickle("embeddings/incidents_with_embeddings.pkl")

# Request body model
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

# Clean text function (reuse your existing one if any)
import re
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9 .,!?]', '', text)
    return text.strip()

def get_embedding(text: str):
    response = client.embeddings.create(
        model=embedding_deployment,
        input=text
    )
    return np.array(response.data[0].embedding).astype('float32')

@app.post("/search-similar-incidents")
async def search_similar_incidents(request: SearchRequest):
    query_text = clean_text(request.query)
    if query_text == "":
        raise HTTPException(status_code=400, detail="Empty query")

    # Get query embedding
    query_embedding = get_embedding(query_text)

    # Search in FAISS index
    distances, indices = index.search(np.array([query_embedding]), request.top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        incident = df.iloc[idx].to_dict()
        # You can exclude large fields like embeddings if you want
        incident.pop('embedding', None)
        results.append({
            "incident": incident,
            "distance": float(dist)
        })

    return {"results": results}


@app.post("/recommend-resolution")
def recommend_resolution(request: SearchRequest):
    query_embedding = get_embedding(request.query)
    D, I = index.search(np.array([query_embedding]), request.top_k)

    recommendations = []
    for i in I[0]:
        if i == -1:
            continue
        incident = df.iloc[i]
        combined_text = incident["combined_text"]
        ticket_number = incident["Ticket Number"]

        # Very basic heuristic to extract resolution-like part
        if "to address" in combined_text.lower():
            part = combined_text.lower().split("to address", 1)[1]
            resolution_text = "To address" + part.split("\n")[0]  # 1st sentence after "To address"
        else:
            resolution_text = combined_text  # fallback

        recommendations.append({
            "ticket": ticket_number,
            "suggested_resolution": resolution_text
        })

    return {"recommendations": recommendations}

