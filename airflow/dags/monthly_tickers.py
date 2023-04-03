from airflow.decorators import dag, task
from datetime import datetime, timedelta

from scrapers import get_listed_options
from google.cloud import storage

default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}


@dag(
    schedule="30 3 1 * *",
    start_date=datetime(2023, 3, 1),
    catchup=False,
    tags=["monthly_tickers"],
    default_args=default_args
)
def upload_monthly_tickers_dag():
    @task()
    def extract_tickers():
        return get_listed_options()
    @task()
    def load_gcs(tickers):
        storage_client = storage.Client()
        bucket = storage_client.bucket('bqtest-asxdata')
        blob = bucket.blob('monthly_tickers.txt')

        blob.upload_from_string('\n'.join(tickers))

    tickers = extract_tickers()
    load_gcs(tickers)

upload_monthly_tickers_dag()



