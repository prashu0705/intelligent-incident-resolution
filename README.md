# Intelligent Incident Resolution with AI Assistance

This project is an Intelligent Incident Resolution system designed to automate and optimize the process of identifying, categorizing, and resolving IT incidents. It leverages machine learning, natural language processing, and AI-powered assistance to streamline incident management workflows.

## Table of Contents
- [Project Overview and Architecture](#project-overview-and-architecture)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Setup and Running Instructions](#setup-and-running-instructions)
- [API Endpoints Documentation](#api-endpoints-documentation)
- [Production Considerations](#production-considerations)
- [Future Enhancements](#future-enhancements)

## Project Overview and Architecture

The system aims to provide an intelligent layer on top of traditional incident management by:
*   Understanding incident descriptions through natural language processing.
*   Automatically suggesting resolutions based on historical data.
*   Identifying trends and clustering incidents to proactively flag potential widespread issues.
*   Offering an AI assistant to guide L1 support engineers.

The main components of the system are:

1.  **FastAPI Backend (`src/main.py`):**
    *   Provides RESTful APIs for all core functionalities.
    *   Handles requests from the frontend and orchestrates interactions with other components like the FAISS index, data store, and Azure AI services.
    *   Includes logic for similarity search, resolution recommendation, clustering, and interfacing with an Azure AI Agent for advanced tasks.

2.  **Angular Frontend (`frontend/incident-ui/`):**
    *   A web-based user interface for interacting with the system.
    *   The primary feature is an AI Assistant chat interface (`AssistantChatComponent`) that allows users to query the system in natural language.
    *   Displays incident details, resolution suggestions, and other relevant information.

3.  **Azure Function (`azure_function_cluster/function_app.py`):**
    *   A serverless function designed for scheduled, background tasks.
    *   Currently implements HDBSCAN clustering on recent incidents to detect emerging patterns and significant increases in cluster sizes.
    *   Can send email alerts if predefined alert conditions are met.

4.  **Data Processing Scripts (`src/`):**
    *   **`preprocess_data.py`**: Cleans raw incident data (e.g., from Excel) and prepares it for embedding generation.
    *   **`generate_embeddings.py`**: Uses Azure OpenAI to generate vector embeddings for the textual content of incidents.
    *   **`build_faiss_index.py`**: Creates a FAISS index from the generated embeddings for efficient similarity searching.

## Key Features

*   **AI-Powered Assistant:** Chat interface for natural language queries regarding incidents.
*   **Similar Incident Search:** Find past incidents similar to a new issue.
*   **Resolution Recommendation:** Suggest potential resolutions based on historically similar cases.
*   **Automated Incident Classification (via AI Agent):** Suggest category, severity, and priority for new incidents.
*   **Trend Analysis & Clustering:** Identify and display trending incident clusters using HDBSCAN.
*   **Proactive Alerting:**
    *   FastAPI endpoint (`/check-proactive-alerts`) for on-demand alert checks.
    *   Scheduled Azure Function for detecting and alerting on significant cluster growth.
*   **Resolution Rewriting & Summarization:** AI-powered tools to clarify and condense resolution notes.

## Tech Stack

*   **Backend:** Python, FastAPI, Uvicorn
*   **Frontend:** TypeScript, Angular, Angular Material
*   **Machine Learning & NLP:**
    *   Vector Embeddings: Azure OpenAI Embeddings
    *   Similarity Search: FAISS (Facebook AI Similarity Search)
    *   Clustering: HDBSCAN
    *   AI Agent: Azure AI Studio (Foundry)
*   **Data Storage (Development):** CSV, Pickle files, JSON (for state)
*   **Serverless (Scheduled Tasks):** Azure Functions
*   **Environment Management:** `dotenv`

## Setup and Running Instructions

Follow these steps to set up and run the project locally:

**1. Prerequisites:**
    *   Python (3.9+)
    *   Node.js (LTS version) and npm (comes with Node.js)
    *   Angular CLI (`npm install -g @angular/cli`)
    *   Azure Functions Core Tools (`npm install -g azure-functions-core-tools@4` or as per Azure documentation)
    *   Access to an Azure OpenAI instance with an embedding model deployed.
    *   Access to Azure AI Studio for agent capabilities (optional, for full functionality).
    *   (Optional, for email alerts from Azure Function) A Gmail account with an "App Password" enabled if using the provided SMTP configuration.

**2. Clone the Repository:**
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

**3. Backend Setup:**
    *   **Create a virtual environment:**
      ```bash
      python -m venv venv
      source venv/bin/activate  # On Windows: venv\Scripts\activate
      ```
    *   **Install Python dependencies:**
      ```bash
      pip install -r requirements.txt
      # If azure_function_cluster has a separate requirements.txt and you're not in its directory:
      # pip install -r azure_function_cluster/requirements.txt
      ```
      *(Note: Ensure `hdbscan` and other necessary libraries are listed in `requirements.txt`)*

    *   **Configure Environment Variables:**
        *   Create a `.env` file in the root directory of the project.
        *   Add the following variables, replacing placeholders with your actual values:
          ```env
          # Azure OpenAI
          AZURE_OPENAI_ENDPOINT="your_azure_openai_endpoint"
          AZURE_OPENAI_API_KEY="your_azure_openai_api_key"
          AZURE_OPENAI_API_VERSION="your_api_version" # e.g., 2023-05-15
          AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME="your_embedding_deployment_name" # e.g., text-embedding-ada-002

          # Azure AI Studio (Foundry Agent - if used by endpoints like auto-classify)
          AZURE_PROJECT_ENDPOINT="your_azure_ai_project_endpoint" # from AI Studio project
          AZURE_AGENT_ID="your_agent_id" # from AI Studio agent deployment

          # Email for Azure Function Alerts (optional)
          SENDER_EMAIL="your_sender_email@gmail.com"
          RECEIVER_EMAIL="your_receiver_email@example.com"
          APP_PASSWORD="your_gmail_app_password"
          ```

    *   **Run the Data Pipeline:** Execute these scripts in order from the root directory:
      ```bash
      python src/preprocess_data.py
      python src/generate_embeddings.py
      python src/build_faiss_index.py
      ```
      *(Ensure your raw data, e.g., `data/incident_data.xlsx`, is present before running `preprocess_data.py`)*

    *   **Start the FastAPI Backend:**
      ```bash
      uvicorn src.main:app --reload --port 8000
      ```
      The API will be accessible at `http://localhost:8000/docs`.

**4. Frontend Setup:**
    *   **Navigate to the frontend directory:**
      ```bash
      cd frontend/incident-ui
      ```
    *   **Install npm dependencies:**
      ```bash
      npm install
      ```
    *   **Serve the Angular application:**
      ```bash
      ng serve
      ```
      The frontend will be accessible at `http://localhost:4200/`.

**5. Azure Function Setup (Local):**
    *   **Navigate to the Azure Function directory:**
      ```bash
      cd azure_function_cluster
      # (If you installed its requirements separately and not from root, do it here)
      # pip install -r requirements.txt
      ```
    *   **Run the Azure Function locally:**
      ```bash
      func start
      ```
      The function will run based on its timer schedule (`0 */5 * * * *` - every 5 minutes by default).

## API Endpoints Documentation

The FastAPI backend provides the following key API endpoints:

**1. Assistant Chat**
   *   **Endpoint:** `POST /ask-assistant`
   *   **Description:** Main interaction point for the AI assistant. Detects intent from the user's question, calls relevant internal APIs for supporting data, and then queries an Azure AI Agent to generate a comprehensive answer.
   *   **Request Body:**
     ```json
     {
       "question": "How to fix issue with VPN connection?",
       "conversation_history": [
         {"role": "user", "content": "My VPN is not working."},
         {"role": "assistant", "content": "Okay, I can help with that. What error message are you seeing?"}
       ],
       "top_k": 3
     }
     ```
     *(`conversation_history` and `top_k` are optional)*
   *   **Response Body:**
     ```json
     {
       "session_id": "generated-uuid-if-not-provided-in-header",
       "answer": "The AI agent's response to the question...",
       "supporting_data": {
         "similar_incidents": { /* ... results from /search-similar-incidents ... */ },
         "recommend_resolutions": { /* ... results from /recommend-resolution ... */ }
         /* ... other data based on intent ... */
       },
       "chat_history": [ /* ... updated full conversation history ... */ ]
     }
     ```

**2. Search Similar Incidents**
   *   **Endpoint:** `POST /search-similar-incidents`
   *   **Description:** Searches for incidents similar to the provided query text using vector similarity.
   *   **Request Body:**
     ```json
     {
       "query": "Cannot print from my desktop.",
       "top_k": 5
     }
     ```
   *   **Response Body:**
     ```json
     {
       "results": [
         {
           "incident": { /* ... incident details (excluding embedding) ... */ },
           "distance": 0.85
         }
         // ... other similar incidents
       ]
     }
     ```

**3. Recommend Resolution**
   *   **Endpoint:** `POST /recommend-resolution`
   *   **Description:** Finds similar past incidents and extracts their resolution notes to suggest potential solutions.
   *   **Request Body:**
     ```json
     {
       "query": "Outlook keeps crashing on startup.",
       "top_k": 3
     }
     ```
   *   **Response Body:**
     ```json
     {
       "recommendations": [
         {
           "ticket": "INC0012345",
           "suggested_resolution": "Recreated Outlook profile, issue resolved.",
           "source_of_resolution": "Extracted from 'Latest Comments' (keyword: 'resolution')."
         }
         // ... other recommendations
       ]
     }
     ```

**4. Get Clusters (Trending Issues)**
   *   **Endpoint:** `GET /get-clusters`
   *   **Description:** Identifies and returns trending incident clusters based on HDBSCAN clustering of recent incidents.
   *   **Request Body:** None
   *   **Response Body:**
     ```json
     {
       "status": "success",
       "clusters": [
         {
           "cluster_id": 0,
           "count": 15,
           "main_title": "Network Printer Access",
           "representative_ticket": { "id": "INC0056789", "category": "Hardware", "severity": "3", "description": "User unable to connect to network printer..." }
         }
         // ... other clusters (top 5)
       ],
       "total_incidents_analyzed": 100,
       "clustered_incidents": 85,
       "timestamp": "2023-10-27T10:30:00.123Z"
     }
     ```

**5. Check Proactive Alerts**
   *   **Endpoint:** `POST /check-proactive-alerts`
   *   **Description:** Checks for conditions that might warrant a proactive alert, such as rapidly growing clusters, a high number of critical tickets, or high recent incident volume.
   *   **Request Body:**
     ```json
     {
       "threshold": 5,
       "lookback_days": 7
     }
     ```
     *(`lookback_days` is not fully utilized in the current implementation but is available for future enhancements)*
   *   **Response Body:**
     ```json
     {
       "alerts": [
         {
           "type": "active_cluster",
           "key": "cluster_0",
           "title": "Network Printer Access",
           "count": 15,
           "example_ticket": "INC0056789",
           "message": "Active cluster: Network Printer Access (15 cases)"
         }
         // ... other alerts
       ],
       "analysis_method": "cluster_based_and_priority",
       "cache_timestamp": "2023-10-27T10:35:00.456Z"
     }
     ```

**6. Auto-Classify Ticket**
   *   **Endpoint:** `POST /auto-classify-ticket`
   *   **Description:** Uses an AI agent to suggest classification (category, severity, priority, assignment group) for an incident based on its description.
   *   **Request Body:**
     ```json
     {
       "query": "User reports that the shared drive is inaccessible. Error message 'Network path not found'."
     }
     ```
   *   **Response Body:**
     ```json
     {
       "classification": "Category: Network, Severity: 2-High, Priority: 2-High, Assignment Group: Network Support L2"
     }
     ```
     *(Actual text will be the AI agent's formatted suggestion)*

## Production Considerations

The current implementation of this system utilizes local storage mechanisms for data and search indexing:

*   **Incident Data:** Stored in local CSV files (e.g., `incident_data.xlsx`, `cleaned_incidents.csv`) and Pickle files (`incidents_with_embeddings.pkl`).
*   **Vector Search:** Employs a local FAISS (Facebook AI Similarity Search) index (`incident_embeddings.index`) for finding similar incidents based on embeddings.

While suitable for development and small-scale testing, a production environment would necessitate more robust, scalable, and managed solutions. For an Azure-based deployment, the following services would be recommended:

*   **Azure Cosmos DB:** This globally distributed, multi-model database service would replace local file storage for incident data.
    *   **Benefits:** High availability, elastic scalability, low-latency access, and robust data management capabilities.
*   **Azure AI Search (formerly Azure Cognitive Search):** This service would replace the local FAISS index for vector search and similarity analysis.
    *   **Benefits:** Managed indexing and querying of vector embeddings, integration with other Azure AI services, built-in security, scalability, and support for hybrid search (combining vector search with keyword-based search).

Transitioning to these managed Azure services would significantly enhance the system's reliability, performance, scalability, and maintainability in a production setting.

## Future Enhancements
*   Integration with live ticketing systems (e.g., ServiceNow, Jira) for data ingestion and updates.
*   More sophisticated state management for the Azure Function alerts (e.g., using Azure Table Storage).
*   User authentication and authorization.
*   CI/CD pipelines for automated deployment.
*   Advanced analytics dashboard for visualizing trends and model performance.
*   Fine-tuning embedding models on domain-specific data.
