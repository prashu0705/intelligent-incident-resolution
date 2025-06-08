import logging
import azure.functions as func
import pandas as pd
import numpy as np
import hdbscan

def main(mytimer: func.TimerRequest) -> None:
    logging.info('ClusterIncidents function started.')

    try:
        # Load incidents with embeddings pickle
        df = pd.read_pickle(project_root / "embeddings" / "incidents_with_embeddings.pkl")

        # Extract embeddings as numpy array
        embeddings = np.vstack(df['embedding'].values)

        # Cluster with HDBSCAN
        cluster_labels = hdbscan.HDBSCAN(min_cluster_size=10).fit_predict(embeddings)

        # Add cluster labels to df
        df['cluster'] = cluster_labels

        # Count how many incidents per cluster
        cluster_freq = df['cluster'].value_counts().sort_index()

        logging.info(f"Cluster frequencies:\n{cluster_freq}")

        # Simple alert if cluster frequency too high (threshold 10)
        ALERT_THRESHOLD = 10
        for cluster_id, freq in cluster_freq.items():
            if freq >= ALERT_THRESHOLD:
                logging.warning(f"ALERT: Cluster {cluster_id} has {freq} incidents!")

    except Exception as e:
        logging.error(f"Error in clustering incidents: {e}")

    logging.info('ClusterIncidents function completed.')

