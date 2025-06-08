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
import hdbscan
from datetime import datetime, timedelta
from typing import Dict, List, Optional

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

# Add cleaned text column if not exists
if 'cleaned_text' not in df.columns:
    df['cleaned_text'] = df['combined_text'].apply(clean_text)

# ------------------ Pydantic Request Models ------------------ #
class SearchRequest(BaseModel):
    query: str
    top_k: int = 5

class AskRequest(BaseModel):
    question: str
    conversation_history: list = None
    top_k: int = 3

class AlertRequest(BaseModel):
    threshold: int = 5
    lookback_days: int = 7

# ------------------ In-Memory Conversation Storage ------------------ #
conversation_memory = {}
alert_cache = {}

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

def detect_query_intent(query: str) -> Dict[str, bool]:
    """Determine which APIs to call based on the user's query"""
    query = query.lower()
    intent = {
        "similar_incidents": False,
        "recommend_resolution": False,
        "auto_classify": False,
        "cluster_analysis": False,
        "proactive_alert": False
    }
    
    similar_phrases = ["similar incidents", "seen this before", "like this before", "past cases"]
    resolution_phrases = ["how to fix", "recommend resolution", "suggest solution", "how was this resolved"]
    classify_phrases = ["what category", "what severity", "what priority", "how should this be classified"]
    cluster_phrases = ["trending issues", "common problems", "frequent incidents", "pattern detection"]
    alert_phrases = ["any alerts", "growing issues", "increasing tickets", "repeated failures"]
    
    intent["similar_incidents"] = any(phrase in query for phrase in similar_phrases)
    intent["recommend_resolution"] = any(phrase in query for phrase in resolution_phrases)
    intent["auto_classify"] = any(phrase in query for phrase in classify_phrases)
    intent["cluster_analysis"] = any(phrase in query for phrase in cluster_phrases)
    intent["proactive_alert"] = any(phrase in query for phrase in alert_phrases)
    
    # Default to similar incidents if no specific intent detected
    if not any(intent.values()):
        intent["similar_incidents"] = True
    
    return intent

def generate_agent_prompt(question: str, api_results: Dict[str, any]) -> str:
    """Generate a comprehensive prompt for the Foundry agent"""
    prompt_parts = [
        "You are an advanced IT support assistant. Use the following information to answer the user's question:",
        f"\nUser Question: {question}"
    ]
    
    if api_results.get("similar_incidents"):
        prompt_parts.append("\n[SIMILAR INCIDENTS FOUND]")
        for incident in api_results["similar_incidents"]["results"][:3]:
            prompt_parts.append(
                f"- Ticket {incident['incident']['Ticket Number']} (similarity: {incident['distance']:.2f}): "
                f"{incident['incident'].get('cleaned_text', '')[:200]}..."
            )
    
    if api_results.get("recommend_resolutions"):
        prompt_parts.append("\n[RECOMMENDED RESOLUTIONS]")
        for res in api_results["recommend_resolutions"]["recommendations"][:3]:
            prompt_parts.append(
                f"- From Ticket {res['ticket']}: {res['suggested_resolution'][:200]}..."
            )
    
    if api_results.get("classification"):
        prompt_parts.append("\n[AUTO-CLASSIFICATION SUGGESTION]")
        prompt_parts.append(api_results["classification"]["classification"])
    
    if api_results.get("clusters"):
        prompt_parts.append("\n[TRENDING ISSUES]")
        for cluster in api_results["clusters"]["clusters"][:3]:
            prompt_parts.append(
                f"- Cluster {cluster['cluster_id']}: {cluster['main_title']} "
                f"({cluster['count']} similar incidents)"
            )
    
    if api_results.get("alerts"):
        prompt_parts.append("\n[ACTIVE ALERTS]")
        for alert in api_results["alerts"]["alerts"][:3]:
            prompt_parts.append(
                f"- Alert on {alert['category']}: {alert['count']} occurrences "
                f"(first detected {alert['first_detected']})"
            )
    
    prompt_parts.append("\n[INSTRUCTIONS]")
    prompt_parts.append(
        "- Provide a concise, actionable response to the user's question "
        "using the context above. Reference specific tickets or patterns when relevant."
        "- If suggesting solutions, indicate which past tickets they're based on."
        "- For trending issues, mention the cluster IDs and ticket counts."
        "- Format your response clearly with bullet points or numbered steps when appropriate."
    )
    
    return "\n".join(prompt_parts)

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
async def recommend_resolution(request: SearchRequest):
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

@app.post("/auto-classify-ticket")
async def auto_classify_ticket(request: SearchRequest):
    prompt = f"""
You are an IT assistant. Based on the following incident description, suggest:
1. Likely category
2. Expected severity (1=critical, 4=low)
3. Suggested priority (1=urgent, 3=normal)
4. Recommended assignment group

Format as a bulleted list with brief justification for each.

Incident: "{request.query}"
"""
    thread = project.agents.threads.create()
    project.agents.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = project.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent_id)

    if run.status == "failed":
        raise HTTPException(status_code=500, detail="Agent run failed")

    messages = list(project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING))
    reply = next(
        (m.text_messages[-1].text.value for m in reversed(messages)
         if m.role == "assistant" and m.text_messages),
        "No response from agent"
    )
    return {"classification": reply}

@app.get("/get-clusters")
async def get_clusters():
    """
    Returns trending issue clusters from the incident data
    """
    try:
        # Get the most recent 100 incidents
        recent_incidents = df.tail(100).copy()
        
        # Check if embeddings exist and are in the correct format
        if 'embedding' not in recent_incidents.columns:
            raise HTTPException(
                status_code=400,
                detail="No embeddings found in dataset"
            )
        
        # Convert embeddings to numpy array if they're stored as strings
        if isinstance(recent_incidents['embedding'].iloc[0], str):
            recent_incidents['embedding'] = recent_incidents['embedding'].apply(lambda x: np.fromstring(x.strip('[]'), sep=' '))
        
        # Prepare embeddings for clustering
        embeddings = np.vstack(recent_incidents['embedding'].values)
        
        # Perform clustering with HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,
            min_samples=2,
            cluster_selection_epsilon=0.5
        )
        cluster_labels = clusterer.fit_predict(embeddings)
        recent_incidents['cluster'] = cluster_labels
        
        # Filter out noise (cluster -1)
        clustered = recent_incidents[recent_incidents['cluster'] != -1]
        
        # Get cluster information
        cluster_info = []
        for cluster_id in clustered['cluster'].unique():
            cluster_tickets = clustered[clustered['cluster'] == cluster_id]
            count = len(cluster_tickets)
            
            # Get most common category as the main title
            if 'Category' in cluster_tickets.columns:
                main_title = cluster_tickets['Category'].value_counts().index[0]
            else:
                # Fallback to cleaned text if category doesn't exist
                main_title = cluster_tickets['cleaned_text'].iloc[0][:50] + "..."
            
            # Get representative ticket
            representative = cluster_tickets.iloc[0].to_dict()
            
            cluster_info.append({
                "cluster_id": int(cluster_id),
                "count": int(count),
                "main_title": main_title,
                "representative_ticket": {
                    "id": representative.get("Ticket Number", "N/A"),
                    "category": representative.get("Category", "Unknown"),
                    "severity": representative.get("Severity", "Unknown"),
                    "description": representative.get("cleaned_text", "")[:200] + "..."
                }
            })
        
        # Sort by cluster size (descending)
        cluster_info.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            "status": "success",
            "clusters": cluster_info[:5],  # Return top 5 clusters
            "total_incidents_analyzed": len(recent_incidents),
            "clustered_incidents": len(clustered),
            "timestamp": pd.Timestamp.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Clustering failed: {str(e)}"
        )

@app.post("/check-proactive-alerts")
async def check_proactive_alerts(request: AlertRequest):
    """
    Proactive alerts using:
    1. Current clusters from /get-clusters endpoint
    2. Priority-based alerts (if Priority field exists)
    3. Recent incident volume (last N incidents)
    """
    try:
        alerts = []
        
        # 1. Get active clusters from your existing endpoint
        cluster_response = await get_clusters()
        if cluster_response.get("clusters"):
            for cluster in cluster_response["clusters"]:
                if cluster["count"] >= request.threshold:
                    alerts.append({
                        "type": "active_cluster",
                        "key": f"cluster_{cluster['cluster_id']}",
                        "title": cluster["main_title"],
                        "count": cluster["count"],
                        "example_ticket": cluster["representative_ticket"]["id"],
                        "message": f"Active cluster: {cluster['main_title']} ({cluster['count']} cases)"
                    })
        
        # 2. Priority-based alerts (if Priority field exists in df)
        if 'Priority' in df.columns:
            high_priority = df[df['Priority'].isin([1, 'Critical', 'High'])]
            if len(high_priority) > 0:
                alerts.append({
                    "type": "high_priority",
                    "key": "critical_tickets",
                    "count": len(high_priority),
                    "example_ticket": high_priority.iloc[0]['Ticket Number'],
                    "message": f"{len(high_priority)} high priority tickets needing attention"
                })
        
        # 3. Recent volume detection (last N incidents)
        recent_volume = min(100, len(df))  # Analyze last 100 incidents max
        if recent_volume >= request.threshold:
            alerts.append({
                "type": "recent_volume",
                "key": "high_traffic",
                "count": recent_volume,
                "example_ticket": df.iloc[-1]['Ticket Number'],
                "message": f"High ticket volume ({recent_volume} recent incidents)"
            })
        
        return {
            "alerts": sorted(alerts, key=lambda x: x["count"], reverse=True)[:10],
            "analysis_method": "cluster_based",
            "cache_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Alert generation failed: {str(e)}"
        )
@app.post("/ask-assistant")
async def ask_assistant(request: AskRequest, req: Request):
    session_id = req.headers.get("x-session-id") or str(uuid4())
    history = conversation_memory.get(session_id, [])

    if request.conversation_history:
        history.extend(request.conversation_history)

    # Detect which APIs to call based on the question
    intent = detect_query_intent(request.question)
    api_results = {}
    
    # Call relevant APIs based on detected intent
    if intent["similar_incidents"]:
        api_results["similar_incidents"] = await search_similar_incidents(
            SearchRequest(query=request.question, top_k=request.top_k)
        )
    
    if intent["recommend_resolution"]:
        api_results["recommend_resolutions"] = await recommend_resolution(
            SearchRequest(query=request.question, top_k=request.top_k)
        )
    
    if intent["auto_classify"]:
        api_results["classification"] = await auto_classify_ticket(
            SearchRequest(query=request.question)
        )
    
    if intent["cluster_analysis"]:
        api_results["clusters"] = await get_clusters()
    
    if intent["proactive_alert"]:
        api_results["alerts"] = await check_proactive_alerts(
            AlertRequest(threshold=5, lookback_days=7)
        )
    
    # Generate comprehensive prompt for the Foundry agent
    agent_prompt = generate_agent_prompt(request.question, api_results)
    
    # Create a new thread for this question
    thread = project.agents.threads.create()

    project.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=agent_prompt
    )

    run = project.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent_id
    )

    if run.status == "failed":
        raise HTTPException(status_code=500, detail=f"Agent run failed: {run.last_error}")

    messages = list(project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING))

    assistant_reply = next(
        (
            m.text_messages[-1].text.value
             for m in reversed(messages)
            if m.role == "assistant" and m.text_messages
        ),
        "No response from agent"
    )

    # Store conversation history
    history.append({"role": "user", "content": request.question})
    history.append({"role": "assistant", "content": assistant_reply})
    conversation_memory[session_id] = history[-100:]  # Keep last 100 turns

    return {
        "session_id": session_id,
        "answer": assistant_reply,
        "supporting_data": api_results,  # Include all API results in response
        "chat_history": history
    }

@app.get("/rewrite-resolution-instructions")
async def rewrite_resolution_instructions(ticket_number: str):
    row = df[df["Ticket Number"] == ticket_number]
    if row.empty:
        raise HTTPException(status_code=404, detail="Ticket not found")

    raw_text = row.iloc[0]["combined_text"]
    prompt = f"""Rewrite the following resolution steps as a numbered list of clear instructions suitable for an L1 IT engineer to follow:\n\n{raw_text}"""

    thread = project.agents.threads.create()
    project.agents.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = project.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent_id)

    if run.status == "failed":
        raise HTTPException(status_code=500, detail="Agent run failed")

    messages = list(project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING))
    reply = next(
        (m.text_messages[-1].text.value for m in reversed(messages)
         if m.role == "assistant" and m.text_messages),
        "No response generated"
    )
    return {"instructions": reply}

@app.get("/generate-resolution-summary")
async def generate_resolution_summary(ticket_number: str):
    row = df[df["Ticket Number"] == ticket_number]
    if row.empty:
        raise HTTPException(status_code=404, detail="Ticket not found")

    raw_text = row.iloc[0]["combined_text"]
    prompt = f"Summarize this IT incident resolution in 2-3 sentences:\n\n{raw_text}"

    thread = project.agents.threads.create()
    project.agents.messages.create(thread_id=thread.id, role="user", content=prompt)
    run = project.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent_id)

    if run.status == "failed":
        raise HTTPException(status_code=500, detail="Agent run failed")

    messages = list(project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING))
    reply = next(
        (m.text_messages[-1].text.value for m in reversed(messages)
         if m.role == "assistant" and m.text_messages),
        "No summary generated"
    )
    return {"summary": reply}

