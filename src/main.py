import re
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import AzureOpenAI
import os
import requests
import json
from dotenv import load_dotenv
from uuid import uuid4
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # React dev server origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Load environment variables
load_dotenv()

# Initialize Azure OpenAI client for embeddings
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Azure Foundry settings
FOUNDRY_ENDPOINT = os.getenv("AZURE_FOUNDY_ENDPOINT")
FOUNDRY_API_KEY = os.getenv("AZURE_FOUNDY_API_KEY")

# Load FAISS index and DataFrame
project_root = Path(__file__).parent.parent  # Goes up from src/ to project root
index_path = project_root / "embeddings" / "incident_embeddings.index"
print(f"Looking for index at: {index_path}")
print(f"File exists: {index_path.exists()}")
index = faiss.read_index(str(index_path))
# index = faiss.read_index("../embeddings/incident_embeddings.index")
df_path = project_root / "embeddings" / "incidents_with_embeddings.pkl"
df = pd.read_pickle(str(df_path))

# df = pd.read_pickle("embeddings/incidents_with_embeddings.pkl")

# Request models
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class AskRequest(BaseModel):
    question: str
    conversation_history: list = None
    top_k: int = 3

# Text cleaning function
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9 .,!?]', '', text)
    return text.strip()

# Embedding generation
def get_embedding(text: str):
    response = client.embeddings.create(
        model=embedding_deployment,
        input=text
    )
    return np.array(response.data[0].embedding).astype('float32')

# Search similar incidents (existing endpoint)
@app.post("/search-similar-incidents")
async def search_similar_incidents(request: SearchRequest):
    query_text = clean_text(request.query)
    if query_text == "":
        raise HTTPException(status_code=400, detail="Empty query")

    query_embedding = get_embedding(query_text)
    distances, indices = index.search(np.array([query_embedding]), request.top_k)

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        incident = df.iloc[idx].to_dict()
        incident.pop('embedding', None)
        results.append({
            "incident": incident,
            "distance": float(dist)
        })

    return {"results": results}

# Recommend resolution (existing endpoint)
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

        if "to address" in combined_text.lower():
            part = combined_text.lower().split("to address", 1)[1]
            resolution_text = "To address" + part.split("\n")[0]
        else:
            resolution_text = combined_text

        recommendations.append({
            "ticket": ticket_number,
            "suggested_resolution": resolution_text
        })

    return {"recommendations": recommendations}

# In-memory conversation memory store (for demo)
conversation_memory = {}

# New endpoint for /ask-assistant with memory and Azure Foundry call
@app.post("/ask-assistant")
async def ask_assistant(request: AskRequest, req: Request):
    # Get or create session_id from header (client must send or first time gets new)
    session_id = req.headers.get("x-session-id")
    if not session_id:
        session_id = str(uuid4())

    # Load previous conversation history for session if any
    history = conversation_memory.get(session_id, [])

    # Append any new conversation history client sent (optional)
    if request.conversation_history:
        history.extend(request.conversation_history)

    # Get similar incidents from FAISS
    query_embedding = get_embedding(request.question)
    _, indices = index.search(np.array([query_embedding]), request.top_k)

    context = []
    for idx in indices[0]:
        if idx == -1:
            continue
        incident = df.iloc[idx]
        context.append({
            "ticket_number": incident.get("Ticket Number", "N/A"),
            "description": incident.get("Description", ""),
            "resolution": incident.get("Resolution", "")
        })

    # System message with incident context
    system_message = {
        "role": "system",
        "content": f"""You are a helpful IT support assistant. Use the following incident history as context:

{json.dumps(context, indent=2)}

Answer the user's question based on this context and your general knowledge.
"""
    }

    # Compose messages: system + history + user question
    messages = [system_message] + history + [{"role": "user", "content": request.question}]

    # Call Azure Foundry API
    headers = {
        "Content-Type": "application/json",
        "api-key": FOUNDRY_API_KEY
    }
    payload = {
        "messages": messages,
        "temperature": 0.7,
        "max_tokens": 500
    }

    try:
        response = requests.post(
            FOUNDRY_ENDPOINT,
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        assistant_reply = result['choices'][0]['message']['content']

        # Save user question and assistant reply for memory (keep last 20 msgs)
        history.append({"role": "user", "content": request.question})
        history.append({"role": "assistant", "content": assistant_reply})
        conversation_memory[session_id] = history[-20:]

        return {
            "session_id": session_id,
            "answer": assistant_reply,
            "context_tickets": [incident["ticket_number"] for incident in context],
            "full_response": result
        }
    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error calling Azure Foundry: {str(e)}")

