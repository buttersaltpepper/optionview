from airflow.decorators import dag, task
from datetime import datetime, timedelta

import scrapers
import pandas as pd
from google.cloud import bigquery
client = bigquery.Client()

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
def ticker_financials_dag():
    @task()
    def get_tickers(**kwargs):
        from google.cloud import storage
        storage_client = storage.Client()

        bucket_name = kwargs['bucket_name']
        blob_name = kwargs['blob_name']
        
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        contents = blob.download_as_string().decode("utf-8")
        return contents.split("\n")

    @task()
    def info_etl(tickers):
        df = scrapers.get_company_directory()

        job_config = bigquery.LoadJobConfig(
            autodetect=True,
            write_disposition='WRITE_TRUNCATE'
            )

        job = client.load_table_from_dataframe(df, 'bqtest-379921.asxdata.companydirectory', job_config=job_config)
        job.result()
        print('Loaded company directory')

    @task()
    def holders_etl(tickers):
        dfs = []
        count = 0
        total = len(tickers)
        for ticker in tickers:
            print(f"Processing for {ticker} ({count+1}/{total})...")
            try:      
                df = scrapers.get_holders(ticker+'.AX')
                if df is not None:
                    df['ticker'] = ticker
                    dfs.append(df)
                    count+=1
            except:
                pass
        
        # Transform
        holders_df = pd.concat(dfs)
        holders_df = holders_df.dropna()

        holders_df['Date reported'] = pd.to_datetime(holders_df['Date reported'])
        holders_df.columns = holders_df.columns.str.lower().str.strip().str.replace(' ', '').str.replace('%','pct')
        holders_df['shares'] = holders_df['shares'].str.replace(',', '').astype(int)
        holders_df['value'] = holders_df['value'].str.replace(',', '').astype(int)
        holders_df['pctout'] = holders_df['pctout'].str.replace('%', '').astype(float)

        # Load
        job_config = bigquery.LoadJobConfig(write_disposition='WRITE_TRUNCATE')
        job = client.load_table_from_dataframe(holders_df, 'bqtest-379921.asxdata.holders', job_config=job_config)
        job.result()
        print('Loaded holders')


    @task()
    def earnings_etl(tickers):
        dfs = []
        count = 0
        total = len(tickers)
        for ticker in tickers:
            print(f"Processing for {ticker} ({count+1}/{total})...")
            try:
                df = scrapers.get_earnings(ticker)
                if df is not None:
                    df['ticker'] = ticker
                    dfs.append(df)
                    count+=1
            except:
                pass

        # Transform
        earnings_df = pd.concat(dfs)
        earnings_df['report_date'] = pd.to_datetime(earnings_df['report_date'])
        earnings_df['forecast'] = earnings_df['forecast'].astype(float)
        earnings_df['eps'] = earnings_df['eps'].astype(float)

        # Load
        job_config = bigquery.LoadJobConfig(
            schema=[
            bigquery.SchemaField('report_date', 'DATE'),
            bigquery.SchemaField('forecast', 'FLOAT64'),
            bigquery.SchemaField('eps', 'FLOAT64'),
            ],
            write_disposition='WRITE_TRUNCATE')

        job = client.load_table_from_dataframe(earnings_df, 'bqtest-379921.asxdata.earnings', job_config=job_config)
        job.result()
        print('Loaded earnings')


    list_tickers = get_tickers(bucket_name='bqtest-asxdata', blob_name='monthly_tickers.txt')
    info_etl(list_tickers)
    holders_etl(list_tickers)
    earnings_etl(list_tickers)

ticker_financials_dag()