# Import libraries
from airflow.decorators import dag, task

import scrapers
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import yfinance as yf
from google.cloud import storage

# For local
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'bqtest-379921-c8ee254b4e40.json'

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


def implied_volatility(row, S):
    market_price = row['price_theoretical']
    def f(sigma):
        return bs_price(S, row['price_exercise'], 0, sigma, (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365, row['option_type']) - market_price
    try:
        return brentq(f, 1e-6, 10)
    except ValueError:
        return np.nan
    

def option_greeks(df):
    # define Black-Scholes formula
    S = yf.download("^AXJO", start= datetime.today().date() - timedelta(days=5))['Close'][-1]
    # calculate Delta, Gamma, and IV for each option in the dataframe
    df['delta'] = df.apply(lambda row: bs_delta(S, row['price_exercise'], 0, 0.2, (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365, row['option_type']), axis=1)
    df['gamma'] = df.apply(lambda row: bs_gamma(S, row['price_exercise'], 0, 0.2, (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365), axis=1)
    # add implied volatility column to dataframe
    df['implied_volatility'] = df.apply(lambda row: implied_volatility(row, S), axis=1)
    df['implied_volatility'] = round(df['implied_volatility'],3)
    df = df.fillna(0)
    return df

def upload_blob_from_memory(bucket_name, contents, destination_blob_name):
    """Uploads a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_string(contents.to_csv(), 'text/csv')

    print(
        f"{destination_blob_name} with contents {contents} uploaded to {bucket_name}."
    )



default_args = {
    "owner": "airflow",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=10),
}

@dag(
    schedule="40 3 * * 1-5",
    start_date=datetime(2023, 3, 1),
    catchup=False,
    tags=["market_etl"],
    default_args=default_args
)

def xjo_optionchain_etl():
    @task()
    def start():
        pass

    @task()
    def etl():
        # Scrape XJO option chains
        df = scrapers.get_option_chain('XJO')

        # Run function to add greeks columns
        df_greeks = option_greeks(df)

        # Use date as name of csv & upload to gcs
        blob_name = str(datetime.today().date()).replace('-','_')+'.csv'
        upload_blob_from_memory('xjo_option_chain', df_greeks, blob_name)

    @task()
    def end():
        pass

    start() >> etl() >> end()

xjo_optionchain_etl()
