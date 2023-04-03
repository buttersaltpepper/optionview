from airflow.decorators import dag, task
from datetime import datetime, timedelta
import scrapers
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf

from google.cloud import bigquery
from google.cloud import storage

# Local only
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/home/buttersaltpepper/airflow/dags/keys/bqtest-379921-c8ee254b4e40.json'

client=bigquery.Client()
storage_client = storage.Client()


def upload_blob_from_memory(bucket_name, contents, destination_blob_name):
    """Uploads a file to the bucket."""
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(contents.to_csv(), 'csv')

    print(
        f"{destination_blob_name} with contents {contents} uploaded to {bucket_name}."
    )


def bs_price(S, K, r, sigma, T, option_type):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'Call':
        price = S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r*T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return price

# define function to calculate theoretical price
def calculate_price(row, S):
    K = row['price_exercise']
    r = 0  # risk-free rate
    T = (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365
    option_type = row['option_type']
    price = bs_price(S, K, r, 0.2, T, option_type)
    return price

def bs_delta(S, K, r, sigma, T, option_type):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    if option_type.lower() == 'call':
        delta = norm.cdf(d1)
    else:
        delta = -norm.cdf(-d1)
    return delta

def bs_gamma(S, K, r, sigma, T):
    d1 = (np.log(S/K) + (r + sigma**2/2) * T) / (sigma * np.sqrt(T))
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    return gamma



default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

@dag(
    schedule="30 10,3 * * 1-5",
    start_date=datetime(2023, 3, 1),
    catchup=False,
    tags=["market_etl"],
    default_args=default_args
)

def market_data_dag():
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
    def optiondata_to_gcs(tickers):
        count = 0
        total = len(tickers)
        result = []
        for ticker in tickers:
            print(f"Processing for {ticker} ({count+1}/{total})...")
            try:
                df = scrapers.get_option_chain(ticker)
                if df is not None:
                    if ticker == 'XJO':
                        S = yf.download("^AXJO", start= datetime.today().date() - timedelta(days=5))['Close'][-1]
                    else:
                        S = yf.download(ticker + '.AX', start= datetime.today().date() - timedelta(days=5))['Close'][-1]
                    df['ticker'] = ticker
                    df['delta'] = df.apply(lambda row: bs_delta(S, row['price_exercise'], 0, 0.2, (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365, row['option_type']), axis=1)
                    df['gamma'] = df.apply(lambda row: bs_gamma(S, row['price_exercise'], 0, 0.2, (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365), axis=1)
                    result.append(df)
                    count+=1
                print(len(df))
            except Exception as e:
                print(e)
                pass
        optiondata = pd.concat(result)

        # Use date as name of csv & upload to gcs
        file_path = 'optionchains/'
        blob_name = str(datetime.today().date()).replace('-','_')+'.csv'
        upload_blob_from_memory('bqtest-asxdata', optiondata, file_path+blob_name)
        return blob_name

    @task()
    def optiondata_to_gbq(blob_name):
        bucket_name = 'bqtest-asxdata/optionchains'
        gcs_uri = f"gs://{bucket_name}/{blob_name}"
        df = pd.read_csv(gcs_uri)
        df = df.iloc[:,1:]
        df['expiration_date'] = pd.to_datetime(df.expiration_date)

        job_config = bigquery.LoadJobConfig(
            schema = [bigquery.SchemaField("expiration_date", "DATE")],
            write_disposition='WRITE_TRUNCATE')
        job = client.load_table_from_dataframe(df, 'bqtest-379921.asxdata.options', job_config=job_config)
        job.result() # Waits for the job to complete.
        print(f"Finished loading options")

    # @task()
    # def etl_marketdata(tickers, default_start=datetime.strptime('2020-01-01', '%Y-%m-%d')):
    #     import yfinance as yf
    #     query = """
    #         SELECT ticker, MAX(date) as max_date
    #         FROM `bqtest-379921.asxdata.marketdata`
    #         WHERE ticker IN ({})
    #         GROUP BY ticker
    #     """.format("'" + "','".join(tickers) + "'")

    #     query_job = client.query(query)
    #     query_job.result()
    #     print(f"Pulled prices from yfinance")

    #     #     # Dictionary of ticker:max(date) from bigquery table
    #     last_updated = {row['ticker']: row['max_date'] for row in query_job}
    #     #     # Create a new dictionary to store ticker:date if not exist
    #     pull_dates = {}
    #     for ticker in tickers:
    #         pull_dates[ticker] = last_updated.get(ticker, default_start)

    #     dfs = []
    #     count = 0
    #     total = len(tickers)
    #     for ticker in tickers:
    #         start_date = pull_dates[ticker]
    #         print(f"Processing for {ticker} ({count+1}/{total})...")
    #         df = yf.download(ticker+'.AX', start=start_date + timedelta(days=1))
    #         df['ticker'] = ticker
    #         dfs.append(df)
    #         count+=1
    #         if ticker == 'XJO':
    #             df = yf.download('^AXJO', start=start_date + timedelta(days=1))
    #             df['ticker'] = ticker
    #             dfs.append(df)
    #             count+=1

    #     if len(dfs) != 0:
    #         marketdata = pd.concat(dfs).drop('Adj Close', axis=1)
    #         marketdata.reset_index(inplace=True)
    #         marketdata.columns = marketdata.columns.str.lower()
    #         marketdata = marketdata.round(2)

    #     if len(marketdata) != 0:
    #         job_config = bigquery.LoadJobConfig()
    #         job_config.write_disposition = bigquery.WriteDisposition.WRITE_APPEND
    #         job = client.load_table_from_dataframe(marketdata, 'bqtest-379921.asxdata.marketdata', job_config=job_config)
    #         job.result()

    list_tickers = get_tickers(bucket_name='bqtest-asxdata', blob_name='monthly_tickers.txt')
    blobname = optiondata_to_gcs(list_tickers)
    optiondata_to_gbq(blobname)
    # etl_marketdata(list_tickers)


market_data_dag()

