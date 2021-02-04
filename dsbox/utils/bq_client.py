import google.auth
from google.cloud import bigquery
from google.cloud import bigquery_storage

SCOPES = ["https://www.googleapis.com/auth/cloud-platform", "https://www.googleapis.com/auth/drive"]


def get_bq_client():
    """
    Build a BigQuery Python client.

    Returns
    -------
    bigquery.Client instance
    """
    credentials, project_id = google.auth.default(
        scopes=SCOPES
    )

    bqclient = bigquery.Client(
        credentials=credentials,
        project=project_id,
    )

    return bqclient


def get_bq_storage_client():
    """
    Build a BigQueryStorage Python client.

    Returns
    -------
    bigquery_storage.BigQueryReadClient instance
    """
    credentials, project_id = google.auth.default(
        scopes=SCOPES
    )

    bqstorageclient = bigquery_storage.BigQueryReadClient(
        credentials=credentials
    )

    return bqstorageclient
