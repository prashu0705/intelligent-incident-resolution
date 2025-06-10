"""
Main FastAPI application for the Intelligent Incident Resolution system.

This application provides several API endpoints for interacting with the system,
including:
- Searching for similar incidents.
- Recommending resolutions based on past incidents.
- Auto-classifying new incidents.
- Retrieving trending issue clusters.
- Checking for proactive alerts.
- An integrated AI assistant that orchestrates calls to other endpoints based on query intent.
- Helper endpoints for rewriting resolution instructions and summarizing resolutions.

It uses a FAISS index for similarity search, HDBSCAN for clustering, and Azure OpenAI
for embeddings and advanced AI agent capabilities.
"""
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
def clean_text(text: str) -> str:
    """
    Cleans input text by converting to lowercase, removing excess whitespace,
    and stripping characters not matching a-z, 0-9, space, comma, period, exclamation, or question mark.

    Args:
        text: The input string to clean.

    Returns:
        The cleaned string. Returns an empty string if input is not a string.
    """
    if not isinstance(text, str):
        return ""
    text = text.lower()  # Convert to lowercase
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single space
    text = re.sub(r'[^a-z0-9 .,!?]', '', text)  # Remove unwanted characters
    return text.strip()  # Remove leading/trailing whitespace

def get_embedding(text: str) -> np.ndarray:
    """
    Generates an embedding for the given text using the configured Azure OpenAI model.

    Args:
        text: The input string for which to generate an embedding.

    Returns:
        np.ndarray: A NumPy array representing the embedding vector (float32).

    Raises:
        Any exception from `client.embeddings.create` if the API call fails.
    """
    response = client.embeddings.create(
        model=embedding_deployment,  # Azure OpenAI embedding deployment name
        input=text
    )
    return np.array(response.data[0].embedding).astype('float32')

def detect_query_intent(query: str) -> Dict[str, bool]:
    """
    Determines the user's intent based on keywords in their query.

    This function checks the query for phrases associated with different actions
    like searching for similar incidents, recommending resolutions, etc.

    Args:
        query: The user's input query string.

    Returns:
        Dict[str, bool]: A dictionary where keys are intent types (e.g., "similar_incidents")
                         and values are booleans indicating if the intent is detected.
                         If no specific intent is found, "similar_incidents" defaults to True.
    """
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
    """
    Generates a comprehensive prompt for the Azure AI Agent based on the user's question
    and the results from various internal API calls (supporting data).

    Args:
        question: The original user question.
        api_results: A dictionary containing results from other API calls like
                     /search-similar-incidents, /recommend-resolution, etc.
                     Example: {"similar_incidents": {...}, "recommend_resolutions": {...}}

    Returns:
        str: A formatted string prompt to be sent to the Azure AI Agent.
    """
    prompt_parts = [
        "You are an advanced IT support assistant. Use the following information to answer the user's question:",
        f"\nUser Question: {question}"
    ]
    
    # Append similar incidents information if available
    if api_results.get("similar_incidents") and api_results["similar_incidents"].get("results"):
        prompt_parts.append("\n[SIMILAR INCIDENTS FOUND]")
        for incident in api_results["similar_incidents"]["results"][:3]:
            prompt_parts.append(
                f"- Ticket {incident['incident']['Ticket Number']} (similarity: {incident['distance']:.2f}): "
                f"{incident['incident'].get('cleaned_text', '')[:200]}..."
            )
    
    # Append recommended resolutions information if available
    if api_results.get("recommend_resolutions") and api_results["recommend_resolutions"].get("recommendations"):
        prompt_parts.append("\n[RECOMMENDED RESOLUTIONS]")
        for res in api_results["recommend_resolutions"]["recommendations"][:3]: # Show top 3
            prompt_parts.append(
                f"- From Ticket {res['ticket']}: {res['suggested_resolution'][:200]}..."
            )
    
    # Append auto-classification suggestion if available
    if api_results.get("classification") and api_results["classification"].get("classification"):
        prompt_parts.append("\n[AUTO-CLASSIFICATION SUGGESTION]")
        prompt_parts.append(api_results["classification"]["classification"])
    
    # Append trending issues (clusters) information if available
    if api_results.get("clusters") and api_results["clusters"].get("clusters"):
        prompt_parts.append("\n[TRENDING ISSUES]")
        for cluster in api_results["clusters"]["clusters"][:3]: # Show top 3
            prompt_parts.append(
                f"- Cluster {cluster['cluster_id']}: {cluster['main_title']} "
                f"({cluster['count']} similar incidents)"
            )
    
    # Append active alerts information if available
    if api_results.get("alerts") and api_results["alerts"].get("alerts"):
        prompt_parts.append("\n[ACTIVE ALERTS]")
        for alert in api_results["alerts"]["alerts"][:3]: # Show top 3
            prompt_parts.append(
                f"- Alert on {alert.get('title', alert.get('category', 'N/A'))}: {alert['count']} occurrences " # Use title if available
                f"(first detected {alert.get('first_detected', 'N/A')})"
            )
    
    # General instructions for the agent
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
@app.post("/search-similar-incidents", summary="Search for similar incidents")
async def search_similar_incidents(request: SearchRequest):
    """
    POST /search-similar-incidents

    Searches for incidents similar to a given query text using vector embeddings
    and a FAISS index.

    Args:
        request (SearchRequest): Pydantic model containing:
            - query (str): The text to search for similar incidents.
            - top_k (int, optional): The number of similar incidents to return. Defaults to 5.

    Returns:
        dict: A JSON object with a "results" key. The value is a list of dictionaries,
              each containing an "incident" (original incident data excluding embedding)
              and its "distance" (similarity score) to the query.
              Example: {"results": [{"incident": {...}, "distance": 0.X}, ...]}

    Raises:
        HTTPException:
            - 400: If the query is empty.
            - 500: If FAISS search or embedding generation fails.
    """
    query_text = clean_text(request.query)
    if not query_text: # Check if empty after cleaning
        raise HTTPException(status_code=400, detail="Query cannot be empty after cleaning.")

    try:
        query_embedding = get_embedding(query_text)
        # FAISS search returns distances and indices
        distances, indices = index.search(np.array([query_embedding]), request.top_k)
    except Exception as e:
        # Log the exception details for server-side debugging
        print(f"Error during embedding generation or FAISS search: {e}") # Or use proper logging
        raise HTTPException(status_code=500, detail="Failed to process query or search similar incidents.")

    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1: # FAISS can return -1 if no more neighbors are found within certain index types
            continue
        incident_data = df.iloc[idx].to_dict()
        incident_data.pop('embedding', None) # Remove embedding from response
        results.append({
            "incident": incident_data,
            "distance": float(dist) # Ensure distance is a standard float
        })

    return {"results": results}

@app.post("/recommend-resolution", summary="Recommend resolutions for an incident query")
async def recommend_resolution(request: SearchRequest):
    """
    POST /recommend-resolution

    Recommends resolutions for a given incident query by finding similar past incidents
    and extracting their resolution information.

    Args:
        request (SearchRequest): Pydantic model containing:
            - query (str): The text describing the incident for which to recommend resolutions.
            - top_k (int, optional): The number of similar incidents to consider. Defaults to 3 in service.

    Returns:
        dict: A JSON object with a "recommendations" key. The value is a list of
              dictionaries, each containing:
                - "ticket" (str): The ticket number of the similar incident.
                - "suggested_resolution" (str): The extracted resolution text.
                - "source_of_resolution" (str): Description of how the resolution was derived.
              Example: {"recommendations": [{"ticket": "T123", "suggested_resolution": "...", "source_of_resolution": "..."}, ...]}

    Raises:
        HTTPException:
            - 400: If the query is empty. (Handled by get_embedding indirectly)
            - 500: If core processing fails.
    """
    query_text = clean_text(request.query)
    if not query_text:
         raise HTTPException(status_code=400, detail="Query cannot be empty after cleaning.")

    try:
        query_embedding = get_embedding(query_text)
        # Search for top_k similar incidents (D: distances, I: indices)
        D, I = index.search(np.array([query_embedding]), request.top_k if request.top_k > 0 else 3)
    except Exception as e:
        print(f"Error during embedding generation or FAISS search for recommendations: {e}")
        raise HTTPException(status_code=500, detail="Failed to process query for recommendations.")

    recommendations = []
    for i in I[0]:
        if i == -1:  # -1 indicates no valid index from FAISS search
            continue
        incident = df.iloc[i]
        ticket_number = incident["Ticket Number"]
        latest_comments = incident.get("Latest Comments", "")
        combined_text = incident.get("combined_text", "") # Fallback

        resolution_text = ""
        extracted_from_keywords = False
        source_of_resolution = "No specific resolution found." # Default message

        if latest_comments and isinstance(latest_comments, str):
            # Try keyword-based extraction from Latest Comments
            keywords = ["resolution:", "resolved by:", "steps taken:", "solution:", "fix:"]
            latest_comments_lower = latest_comments.lower()
            for keyword in keywords:
                if keyword in latest_comments_lower:
                    # Take text after the keyword
                    resolution_text = latest_comments[latest_comments_lower.find(keyword) + len(keyword):].strip()
                    # Further clean up: often resolution is followed by user/date stamps, try to remove them if they are on new lines
                    resolution_lines = [line.strip() for line in resolution_text.splitlines() if line.strip()]
                    # Heuristic: if a line looks like a signature/timestamp, stop before it
                    meaningful_lines = []
                    for line in resolution_lines:
                        if re.match(r"^-+\s*original message\s*-+", line.lower()) or \
                           re.match(r"from:", line.lower()) or \
                           re.match(r"sent:", line.lower()) or \
                           re.match(r"to:", line.lower()) or \
                           re.match(r"cc:", line.lower()) or \
                           re.match(r"subject:", line.lower()) or \
                           re.match(r"\w+\s*,\s*\w+\s+\d+,\s*\d{4}\s+\d+:\d+:\d+\s*(am|pm)", line.lower()) or \
                           re.match(r"\d{1,2}/\d{1,2}/\d{2,4}\s+\d{1,2}:\d{2}\s*(am|pm)?", line.lower()):
                            break
                        meaningful_lines.append(line)
                    resolution_text = "\n".join(meaningful_lines).strip()
                    if resolution_text: # Ensure we still have text after filtering
                        extracted_from_keywords = True
                        source_of_resolution = f"Extracted from 'Latest Comments' (keyword: '{keyword.strip(':')}')."
                    break

            # If no keywords found in Latest Comments, or keyword extraction yielded empty string, try taking last few lines
            if not extracted_from_keywords and latest_comments: # ensure latest_comments is not empty
                lines = [line.strip() for line in latest_comments.splitlines() if line.strip()]
                if lines: # Check if there are any non-empty lines
                    # Heuristic: Avoid taking just "Thank you" or similar short, non-informative lines as resolution
                    candidate_lines = lines[-3:] # Take up to last 3 non-empty lines
                    meaningful_text = "\n".join(candidate_lines)
                    if len(meaningful_text) > 50 or len(candidate_lines) > 1 : # Simple check for meaningful content
                         resolution_text = meaningful_text
                         source_of_resolution = "Extracted from last few lines of 'Latest Comments'."

        # Fallback if no resolution text from Latest Comments
        if not resolution_text.strip():
            if combined_text and isinstance(combined_text, str):
                resolution_text = combined_text # Use full combined_text as last resort for now
                source_of_resolution = "Using 'combined_text' as fallback."
            else: # If combined_text is also empty or not a string
                resolution_text = "No textual information available for this incident."
                source_of_resolution = "No textual information available."

        # Final cleanup and length limiting
        if not resolution_text.strip() or resolution_text == combined_text and not extracted_from_keywords:
            # If we fell back to combined_text and didn't find keywords, it might not be a focused resolution
            # Or if resolution_text is still empty
            if combined_text and isinstance(combined_text, str) and combined_text.strip():
                 resolution_text = f"Review ticket for full details. Summary: {combined_text[:300].strip()}..."
                 source_of_resolution = "General information from 'combined_text'. Review ticket for specifics."
            else:
                 resolution_text = "No specific resolution found in comments; review ticket for full details."
                 source_of_resolution = "No specific resolution text found."

        elif len(resolution_text) > 500: # Arbitrary limit for conciseness
            resolution_text = resolution_text[:497] + "..."
            source_of_resolution += " (Trimmed for brevity)."

        recommendations.append({
            "ticket": ticket_number,
            "suggested_resolution": resolution_text.strip(),
            "source_of_resolution": source_of_resolution
        })

    return {"recommendations": recommendations}

@app.post("/auto-classify-ticket", summary="Automatically classify an incident ticket")
async def auto_classify_ticket(request: SearchRequest):
    """
    POST /auto-classify-ticket

    Uses an Azure AI Agent to suggest a classification for a new incident ticket
    based on its description (query).

    Args:
        request (SearchRequest): Pydantic model containing:
            - query (str): The incident description to classify.
            - top_k is not used by this endpoint but is part of SearchRequest.

    Returns:
        dict: A JSON object with a "classification" key. The value is a string
              containing the agent's suggested classification (e.g., category,
              severity, priority, assignment group).
              Example: {"classification": "Category: Network, Severity: 3, ..."}

    Raises:
        HTTPException:
            - 500: If the AI agent run fails.
    """
    # Prompt engineered for classification task
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
    Returns trending issue clusters from the incident data.

    This endpoint performs HDBSCAN clustering on the most recent 100 incidents
    to identify current trends and patterns.

    Returns:
        dict: A JSON object containing:
            - "status" (str): "success" or error message.
            - "clusters" (List[dict]): A list of the top 5 clusters, each with:
                - "cluster_id" (int): The ID of the cluster.
                - "count" (int): Number of incidents in the cluster.
                - "main_title" (str): A representative title for the cluster (e.g., most common category).
                - "representative_ticket" (dict): Details of a sample ticket from the cluster.
            - "total_incidents_analyzed" (int): Number of incidents processed for clustering.
            - "clustered_incidents" (int): Number of incidents assigned to a valid cluster (excluding noise).
            - "timestamp" (str): ISO format timestamp of when the analysis was run.

    Raises:
        HTTPException:
            - 400: If embeddings are not found or in an invalid format.
            - 500: If the clustering process fails for other reasons.
    """
    try:
        # Get the most recent 100 incidents for trend analysis
        recent_incidents_df = df.tail(100).copy()
        
        if 'embedding' not in recent_incidents_df.columns:
            raise HTTPException(status_code=400, detail="Embedding column not found in dataset.")
        
        # Ensure embeddings are in the correct numerical format for HDBSCAN
        # This check is important if data might come from sources where embeddings are stringified
        if not recent_incidents_df.empty and isinstance(recent_incidents_df['embedding'].iloc[0], str):
            try:
                # Attempt to convert stringified embeddings (e.g., "[0.1, 0.2, ...]") to numpy arrays
                recent_incidents_df['embedding'] = recent_incidents_df['embedding'].apply(
                    lambda x: np.array(ast.literal_eval(x), dtype='float32') if isinstance(x, str) else x
                )
            except (ValueError, SyntaxError) as e:
                raise HTTPException(status_code=400, detail=f"Error converting string embeddings to arrays: {e}")
        
        # Prepare embeddings for HDBSCAN
        # np.vstack requires elements to be arrays/lists of numbers.
        valid_embeddings = recent_incidents_df['embedding'].dropna().tolist()
        if not valid_embeddings:
             raise HTTPException(status_code=400, detail="No valid embeddings available for clustering after processing.")

        embeddings_matrix = np.vstack(valid_embeddings).astype('float32')
        
        # Perform clustering using HDBSCAN
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=5,    # Minimum number of incidents to form a cluster
            min_samples=2,         # Minimum samples in a neighborhood for a point to be a core point
            cluster_selection_epsilon=0.5 # Distance threshold for merging clusters
        )
        cluster_labels = clusterer.fit_predict(embeddings_matrix)
        
        # Assign cluster labels back to the DataFrame subset
        # Only assign to rows that had valid embeddings and were used in clustering
        recent_incidents_df.loc[recent_incidents_df['embedding'].dropna().index, 'cluster'] = cluster_labels
        
        # Filter out noise points (HDBSCAN labels noise as -1)
        clustered_incidents_df = recent_incidents_df[recent_incidents_df['cluster'] != -1]
        
        cluster_info_list = []
        if not clustered_incidents_df.empty:
            for cluster_id_val in clustered_incidents_df['cluster'].unique():
                cluster_tickets_df = clustered_incidents_df[clustered_incidents_df['cluster'] == cluster_id_val]
                count = len(cluster_tickets_df)

                # Determine a main title for the cluster
                main_title_str = "N/A"
                if 'Category' in cluster_tickets_df.columns and not cluster_tickets_df['Category'].value_counts().empty:
                    main_title_str = cluster_tickets_df['Category'].value_counts().index[0]
                elif 'cleaned_text' in cluster_tickets_df.columns and not cluster_tickets_df['cleaned_text'].empty:
                    # Fallback to a snippet of cleaned_text if Category is not informative
                    main_title_str = cluster_tickets_df['cleaned_text'].iloc[0][:50] + "..."

                # Get a representative ticket from the cluster
                representative_ticket_dict = cluster_tickets_df.iloc[0].to_dict()

                cluster_info_list.append({
                    "cluster_id": int(cluster_id_val),
                    "count": int(count),
                    "main_title": main_title_str,
                    "representative_ticket": { # Provide key details of a sample ticket
                        "id": representative_ticket_dict.get("Ticket Number", "N/A"),
                        "category": representative_ticket_dict.get("Category", "Unknown"),
                        "severity": representative_ticket_dict.get("Severity", "Unknown"),
                        "description": representative_ticket_dict.get("cleaned_text", "")[:200] + "..."
                    }
                })

        # Sort clusters by size (descending)
        cluster_info_list.sort(key=lambda x: x['count'], reverse=True)
        
        return {
            "status": "success",
            "clusters": cluster_info_list[:5],  # Return top 5 largest clusters
            "total_incidents_analyzed": len(recent_incidents_df),
            "clustered_incidents": len(clustered_incidents_df), # Count of non-noise incidents
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException: # Re-raise HTTPExceptions directly
        raise
    except Exception as e: # Catch other unexpected errors
        # Log the full error for debugging
        print(f"Unexpected error in /get-clusters: {e}") # Or use proper logging
        raise HTTPException(status_code=500, detail=f"Clustering process failed: {str(e)}")

@app.post("/check-proactive-alerts", summary="Check for proactive alerts based on incident trends")
async def check_proactive_alerts(request: AlertRequest):
    """
    POST /check-proactive-alerts

    Checks for proactive alerts based on several criteria:
    1. Active clusters with incident counts exceeding a threshold (from /get-clusters).
    2. Number of high-priority tickets.
    3. Recent incident volume.

    Args:
        request (AlertRequest): Pydantic model containing:
            - threshold (int): The minimum count for a cluster or volume to trigger an alert.
            - lookback_days (int): Not directly used in this version, but could be for time-based alerts.

    Returns:
        dict: A JSON object containing:
            - "alerts" (List[dict]): A list of active alerts, sorted by count. Each alert includes:
                - "type" (str): e.g., "active_cluster", "high_priority", "recent_volume".
                - "key" (str): A unique key for the alert.
                - "title" (str, optional): Title of the cluster if applicable.
                - "count" (int): The count associated with the alert.
                - "example_ticket" (str, optional): An example ticket ID.
                - "message" (str): A descriptive message for the alert.
            - "analysis_method" (str): Indicates the method used (currently "cluster_based").
            - "cache_timestamp" (str): ISO format timestamp of the alert check.

    Raises:
        HTTPException:
            - 500: If alert generation fails.
    """
    try:
        alerts_list = []

        # 1. Check for active clusters exceeding the threshold
        # This involves calling the /get-clusters endpoint logic internally or directly
        # For simplicity here, we'll assume a conceptual call or direct use of its logic.
        # Note: In a real scenario, avoid await get_clusters() if it's not an async function or if this creates deadlocks.
        # This example assumes get_clusters can be called as if it's providing data.
        # To avoid issues, it might be better to refactor get_clusters logic into a shared function.
        # For this pass, we'll simulate by calling it if it's safe or assume its data structure.
        # The `await get_clusters()` was in the original code; ensure FastAPI handles this correctly if get_clusters is not async.
        # If get_clusters is not async, then `await` should be removed.
        # Given `get_clusters` is `async def`, `await` is correct.
        
        cluster_data = await get_clusters() # Assuming get_clusters is defined as async
        if cluster_data and cluster_data.get("clusters"):
            for cluster_item in cluster_data["clusters"]:
                if cluster_item["count"] >= request.threshold:
                    alerts_list.append({
                        "type": "active_cluster",
                        "key": f"cluster_{cluster_item['cluster_id']}",
                        "title": cluster_item["main_title"],
                        "count": cluster_item["count"],
                        "example_ticket": cluster_item["representative_ticket"]["id"] if cluster_item.get("representative_ticket") else "N/A",
                        "message": f"Active cluster: {cluster_item['main_title']} ({cluster_item['count']} cases)"
                    })
        
        # 2. Check for high-priority tickets (example: Priority 1 or 'Critical')
        if 'Priority' in df.columns:
            # Ensure we handle both numeric and string representations of priority
            critical_priority_values = [1, '1', 'Critical', 'High']
            # Convert Priority column to string for consistent comparison, handling potential NaNs
            high_priority_tickets_df = df[df['Priority'].astype(str).isin([str(p) for p in critical_priority_values])]
            if not high_priority_tickets_df.empty:
                alerts_list.append({
                    "type": "high_priority",
                    "key": "critical_tickets",
                    "count": len(high_priority_tickets_df),
                    "example_ticket": high_priority_tickets_df.iloc[0]['Ticket Number'] if 'Ticket Number' in high_priority_tickets_df.columns else "N/A",
                    "message": f"{len(high_priority_tickets_df)} high priority tickets needing attention."
                })
        
        # 3. Check for recent high volume of incidents (based on the last 100 incidents)
        # This is a simple check on the size of the recent_incidents_df used by /get-clusters
        # or a similar slice of the main DataFrame.
        recent_volume_count = min(100, len(df)) # Based on the same logic as /get-clusters for recent incidents
        if recent_volume_count >= request.threshold and not df.empty:
            alerts_list.append({
                "type": "recent_volume",
                "key": "high_traffic",
                "count": recent_volume_count,
                "example_ticket": df.iloc[-1]['Ticket Number'] if 'Ticket Number' in df.columns else "N/A",
                "message": f"High ticket volume detected ({recent_volume_count} incidents in recent analysis window)."
            })
        
        return {
            "alerts": sorted(alerts_list, key=lambda x: x["count"], reverse=True)[:10], # Return top 10 alerts
            "analysis_method": "cluster_based_and_priority", # Updated method description
            "cache_timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        print(f"Error during proactive alert generation: {e}") # Or use proper logging
        raise HTTPException(status_code=500, detail=f"Proactive alert generation failed: {str(e)}")

@app.post("/ask-assistant", summary="Interact with the AI assistant")
async def ask_assistant(request: AskRequest, req: Request):
    """
    POST /ask-assistant

    Handles user queries by detecting intent, calling relevant internal APIs for data,
    generating a prompt for an Azure AI Agent, and returning the agent's response.
    Manages conversation history using a session ID.

    Args:
        request (AskRequest): Pydantic model containing:
            - question (str): The user's query.
            - conversation_history (list, optional): List of previous turns in the conversation.
              Each turn is a dict like {"role": "user/assistant", "content": "..."}.
            - top_k (int, optional): Number of results for underlying searches (e.g., similar incidents).
        req (Request): The FastAPI request object, used to get headers like "x-session-id".


    Returns:
        dict: A JSON object containing:
            - "session_id" (str): The session ID for the conversation.
            - "answer" (str): The AI agent's textual response.
            - "supporting_data" (dict): Data retrieved from internal APIs, used by the agent.
              (e.g., {"similar_incidents": ..., "recommend_resolutions": ...})
            - "chat_history" (list): The updated conversation history.

    Raises:
        HTTPException:
            - 500: If the AI agent run fails or other internal processing errors occur.
    """
    # Use provided session ID or generate a new one
    session_id = req.headers.get("x-session-id") or str(uuid4())
    # Retrieve conversation history for this session, or start new if not found
    current_conversation_history = conversation_memory.get(session_id, [])

    # Append any history passed in the current request (e.g., if client manages some state)
    if request.conversation_history:
        current_conversation_history.extend(request.conversation_history)

    # Detect user's intent from the question
    intent = detect_query_intent(request.question)
    api_results = {}
    
    # Call relevant APIs based on detected intent to gather supporting data
    # This uses a simple if-chain; more complex routing could be used for many intents.
    if intent["similar_incidents"]:
        try:
            api_results["similar_incidents"] = await search_similar_incidents(
                SearchRequest(query=request.question, top_k=request.top_k)
            )
        except HTTPException as e: # Capture HTTPExceptions from sub-calls to inform the agent
            api_results["similar_incidents_error"] = str(e.detail)

    if intent["recommend_resolution"]:
        try:
            api_results["recommend_resolutions"] = await recommend_resolution(
                SearchRequest(query=request.question, top_k=request.top_k)
            )
        except HTTPException as e:
            api_results["recommend_resolutions_error"] = str(e.detail)
    
    if intent["auto_classify"]:
        try:
            api_results["classification"] = await auto_classify_ticket(
                SearchRequest(query=request.question) # top_k not relevant here
            )
        except HTTPException as e:
            api_results["classification_error"] = str(e.detail)
    
    if intent["cluster_analysis"]:
        try:
            api_results["clusters"] = await get_clusters()
        except HTTPException as e:
            api_results["clusters_error"] = str(e.detail)
    
    if intent["proactive_alert"]:
        try:
            # Default threshold and lookback_days for proactive alerts triggered via chat
            api_results["alerts"] = await check_proactive_alerts(
                AlertRequest(threshold=3, lookback_days=7) # Example default values
            )
        except HTTPException as e:
            api_results["alerts_error"] = str(e.detail)
    
    # Generate a comprehensive prompt for the Azure AI Agent using the gathered API results
    agent_prompt_text = generate_agent_prompt(request.question, api_results)
    
    # Interact with Azure AI Agent (Foundry)
    # Create a new thread for this interaction to keep it isolated
    thread = project.agents.threads.create()

    project.agents.messages.create(
        thread_id=thread.id,
        role="user",
        content=agent_prompt_text
    )

    run = project.agents.runs.create_and_process(
        thread_id=thread.id,
        agent_id=agent_id
    )

    if run.status == "failed":
        raise HTTPException(status_code=500, detail=f"Agent run failed: {run.last_error}")

    messages = list(project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING))

    assistant_reply_text = next(
        (
            m.text_messages[-1].text.value
             for m in reversed(messages)
            if m.role == "assistant" and m.text_messages
        ),
        "No response from agent"
    )

    # Store the updated conversation history (user query + assistant reply)
    current_conversation_history.append({"role": "user", "content": request.question})
    current_conversation_history.append({"role": "assistant", "content": assistant_reply_text})
    # Limit stored history to the last 100 turns to prevent memory overflow
    conversation_memory[session_id] = current_conversation_history[-100:]

    return {
        "session_id": session_id,
        "answer": assistant_reply_text,
        "supporting_data": api_results,  # Return the data used by the agent for transparency/frontend use
        "chat_history": current_conversation_history
    }

@app.get("/rewrite-resolution-instructions", summary="Rewrite resolution instructions for clarity")
async def rewrite_resolution_instructions(ticket_number: str):
    """
    GET /rewrite-resolution-instructions

    Takes a ticket number, retrieves its "combined_text" (summary + comments),
    and uses an Azure AI Agent to rewrite the resolution steps into a clear,
    numbered list suitable for an L1 IT engineer.

    Args:
        ticket_number (str): The ticket number to process.

    Returns:
        dict: A JSON object with an "instructions" key, containing the
              rewritten resolution steps as a string.
              Example: {"instructions": "1. Do this.\n2. Do that.\n..."}

    Raises:
        HTTPException:
            - 404: If the ticket number is not found.
            - 500: If the AI agent run fails.
    """
    # Find the incident by ticket number
    incident_row = df[df["Ticket Number"] == ticket_number]
    if incident_row.empty:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_number} not found.")

    # Use 'combined_text' as the source for resolution information
    raw_resolution_text = incident_row.iloc[0].get("combined_text", "")
    if not raw_resolution_text.strip():
        return {"instructions": "No resolution text found for this ticket to rewrite."}

    # Define the prompt for the AI agent
    rewrite_prompt = f"""Rewrite the following resolution steps as a numbered list of clear instructions suitable for an L1 IT engineer to follow. If the text does not contain clear steps, indicate that.

Original text:
{raw_resolution_text}
"""
    # Interact with Azure AI Agent
    thread = project.agents.threads.create()
    project.agents.messages.create(thread_id=thread.id, role="user", content=rewrite_prompt)
    agent_run = project.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent_id)

    if agent_run.status == "failed":
        # Log the specific error from the agent run if available
        error_detail = f"Agent run failed for rewriting instructions. Last error: {agent_run.last_error}"
        print(error_detail) # Or use proper logging
        raise HTTPException(status_code=500, detail=error_detail)

    # Retrieve the agent's response
    response_messages = list(project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING))
    rewritten_instructions = next(
        (msg.text_messages[-1].text.value for msg in reversed(response_messages)
         if msg.role == "assistant" and msg.text_messages),
        "No instructions generated by the agent."
    )
    return {"instructions": rewritten_instructions}

@app.get("/generate-resolution-summary", summary="Generate a concise summary of an incident resolution")
async def generate_resolution_summary(ticket_number: str):
    """
    GET /generate-resolution-summary

    Takes a ticket number, retrieves its "combined_text", and uses an Azure AI Agent
    to generate a 2-3 sentence summary of the resolution.

    Args:
        ticket_number (str): The ticket number to summarize.

    Returns:
        dict: A JSON object with a "summary" key, containing the resolution summary string.
              Example: {"summary": "The issue was resolved by..."}

    Raises:
        HTTPException:
            - 404: If the ticket number is not found.
            - 500: If the AI agent run fails.
    """
    # Find the incident by ticket number
    incident_row = df[df["Ticket Number"] == ticket_number]
    if incident_row.empty:
        raise HTTPException(status_code=404, detail=f"Ticket {ticket_number} not found.")

    # Use 'combined_text' as the source for resolution information
    raw_resolution_text = incident_row.iloc[0].get("combined_text", "")
    if not raw_resolution_text.strip():
        return {"summary": "No resolution text found for this ticket to summarize."}

    # Define the prompt for the AI agent
    summary_prompt = f"Summarize this IT incident resolution in 2-3 sentences:\n\n{raw_resolution_text}"

    # Interact with Azure AI Agent
    thread = project.agents.threads.create()
    project.agents.messages.create(thread_id=thread.id, role="user", content=summary_prompt)
    agent_run = project.agents.runs.create_and_process(thread_id=thread.id, agent_id=agent_id)

    if agent_run.status == "failed":
        error_detail = f"Agent run failed for generating summary. Last error: {agent_run.last_error}"
        print(error_detail) # Or use proper logging
        raise HTTPException(status_code=500, detail=error_detail)

    # Retrieve the agent's response
    response_messages = list(project.agents.messages.list(thread_id=thread.id, order=ListSortOrder.ASCENDING))
    resolution_summary = next(
        (msg.text_messages[-1].text.value for msg in reversed(response_messages)
         if msg.role == "assistant" and msg.text_messages),
        "No summary generated by the agent."
    )
    return {"summary": resolution_summary}

