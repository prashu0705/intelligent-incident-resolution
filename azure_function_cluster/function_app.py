"""
Azure Function for scheduled incident clustering and alert generation.

This Function App is triggered on a timer schedule. It performs the following:
1. Loads incident data with pre-computed embeddings from a pickle file.
2. Clusters recent incidents using HDBSCAN to identify emerging trends.
3. Compares current cluster sizes with previously stored state.
4. If significant growth is detected in any cluster, an email alert is sent.
5. Saves the current cluster state (including counts and representative details) for the next run.

Environment variables for email (SENDER_EMAIL, RECEIVER_EMAIL, APP_PASSWORD)
must be configured in the Azure Function App settings or local .env file.
The .env file is expected to be in the root directory of the project.
"""
import azure.functions as func
import logging
import pandas as pd
import pickle
import os
from sklearn.cluster import KMeans
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv
from pathlib import Path

app = func.FunctionApp()

# Locate root directory (parent of azure_function_cluster)
root_dir = Path(__file__).resolve().parent.parent

# Load .env from root directory
load_dotenv(dotenv_path=root_dir / ".env")

DATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "embeddings", "incidents_with_embeddings.pkl")
)

STATE_FILE = os.path.join(os.path.dirname(__file__), "cluster_state.json")

def load_incidents():
    with open(DATA_PATH, "rb") as f:
        df = pickle.load(f)
    return df

def save_cluster_state(state):
    with open(STATE_FILE, "w") as f:
        json.dump(state, f)

def load_cluster_state():
    if not os.path.exists(STATE_FILE):
        return {}
    with open(STATE_FILE, "r") as f:
        return json.load(f)

def send_email_alert(subject, body):
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    app_password = os.getenv("APP_PASSWORD")

    if not all([sender_email, receiver_email, app_password]):
        logging.error("Email credentials not found in environment variables")
        return

    msg = MIMEMultipart()
    msg["From"] = sender_email
    msg["To"] = receiver_email
    msg["Subject"] = subject

    msg.attach(MIMEText(body, "plain"))

    try:
        server = smtplib.SMTP_SSL("smtp.gmail.com", 465)
        server.login(sender_email, app_password)
        server.sendmail(sender_email, receiver_email, msg.as_string())
        server.quit()
        logging.info("Alert email sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send email: {str(e)}")

@app.timer_trigger(schedule="0 */5 * * * *", arg_name="myTimer", run_on_startup=True, use_monitor=False)
def RootCauseDetection(myTimer: func.TimerRequest) -> None:
    if myTimer.past_due:
        logging.warning("The timer is past due!")

    logging.info("RootCauseDetection started")

    df = load_incidents()

    embeddings = df['embedding'].tolist()

    n_clusters = 5
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    labels = kmeans.fit_predict(embeddings)

    df['cluster'] = labels

    cluster_counts = df['cluster'].value_counts().to_dict()
    logging.info(f"Current cluster counts: {cluster_counts}")

    prev_counts = load_cluster_state()

    alerts = []
    for cluster_id, count in cluster_counts.items():
        # Get the data for the cluster_id from prev_counts
        prev_data_for_cluster = prev_counts.get(str(cluster_id))

        prev_count = 0 # Default previous count
        if isinstance(prev_data_for_cluster, int):
            prev_count = prev_data_for_cluster
        # If prev_data_for_cluster is not an int (e.g. dict from an old format, None, etc.),
        # prev_count remains 0. This handles the TypeError if it was e.g. a dict.
        
        if count > prev_count + 3:
            alerts.append(f"Cluster {cluster_id} increased from {prev_count} to {count} incidents")

    save_cluster_state({str(k): v for k, v in cluster_counts.items()})

    if alerts:
        for alert in alerts:
            logging.warning(f"ALERT: {alert}")
            send_email_alert("⚠️ Incident Cluster Alert", alert)
    else:
        logging.info("No significant cluster growth detected.")

    logging.info("RootCauseDetection completed")


