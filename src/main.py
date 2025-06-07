import re
import faiss
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from openai import AzureOpenAI
import os
import json
from dotenv import load_dotenv
from uuid import uuid4
from pathlib import Path
from fastapi.middleware.cors import CORSMiddleware

# Foundry SDK imports
from azure.ai.projects import AIProjectClient
from azure.identity import DefaultAzureCredential
from azure.ai.agents.models import ListSortOrder

# ------------------ FastAPI + CORS ------------------ #
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ------------------ Load Env & Init Clients ------------------ #
load_dotenv()

client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION")
)
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME")

# Foundry Agent via SDK
project = AIProjectClient(
    credential=DefaultAzureCredential(),
    endpoint=os.getenv("AZURE_PROJECT_ENDPOINT")
)
agent_id = os.getenv("AZURE_AGENT_ID")

# ------------------ Load FAISS + Incident Data ------------------ #
project_root = Path(__file__).parent.parent
index = faiss.read_index(str(project_root / "embeddings" / "incident_embeddings.index"))
df = pd.read_pickle(str(project_root / "embeddings" / "incidents_with_embeddings.pkl"))

# ------------------ Pydantic Request Models ------------------ #
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class AskRequest(BaseModel):
    question: str
    conversation_history: list = None
    top_k: int = 3

# ------------------ Helper Functions ------------------ #
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

# ------------------ Endpoints ------------------ #

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

# In-memory session memory
conversation_memory = {}

@app.post("/ask-assistant")
async def ask_assistant(request: AskRequest, req: Request):
    session_id = req.headers.get("x-session-id") or str(uuid4())
    history = conversation_memory.get(session_id, [])
    if request.conversation_history:
        history.extend(request.conversation_history)

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

    system_context = f"""You are a helpful IT support assistant. Use this incident history as context:\n\n{json.dumps(context, indent=2)}"""

    # New thread each time (for now)
    thread = project.agents.threads.create()

    project.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=f"{request.question}\n\n[Context]\n{system_context}"
    )

    run = project.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent_id
    )

    if run.status == "failed":
        raise HTTPException(status_code=500, detail=f"Agent run failed: {run.last_error}")

    messages = project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING)
    assistant_reply = next(
        (m.text_messages[-1].text.value for m in reversed(messages) if m.role == "assistant" and m.text_messages),
        "No response from agent"
    )

    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": assistant_reply})
    conversation_memory[session_id] = history[-20:]

    return {
        "session_id": session_id,
        "answer": assistant_reply,
        "context_tickets": [incident["ticket_number"] for incident in context],
    }

