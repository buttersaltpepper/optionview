import requests
from requests_html import HTMLSession
from selectolax.parser import HTMLParser
from datetime import datetime, timedelta
import json
import pandas as pd
from typing import Optional
from dataclasses import dataclass


# Scrape ASX options traded (updated monthly on ASX)
def get_listed_options():
    base_url = 'https://www2.asx.com.au'
    url = "https://www2.asx.com.au/markets/trade-our-derivatives-market/overview/equity-derivatives/equity-derivatives-statistics"
    r = requests.get(url)
    html =  HTMLParser(r.text)

    html.css_first('title').text()
    link = html.css_first('div.button a').attributes['href']

    tickers = pd.read_excel(base_url+link, header=2)['Product'].dropna().tolist()
    tickers = [t for t in tickers if len(t) ==3]
    return tickers


# Scraping ASX.com for option chains
def get_data(ticker):
    # Headers
    headers = {
        'Accept': 'application/json, text/plain, */*',
        'Accept-Language': 'en-US,en;q=0.9',
        'Authorization': 'Bearer 83ff96335c2d45a094df02a206a39ff4',
        'Connection': 'keep-alive',
        'Origin': 'https://www2.asx.com.au',
        'Referer': 'https://www2.asx.com.au/',
        'Sec-Fetch-Dest': 'empty',
        'Sec-Fetch-Mode': 'cors',
        'Sec-Fetch-Site': 'cross-site',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36',
        'sec-ch-ua': '"Chromium";v="110", "Not A(Brand";v="24", "Google Chrome";v="110"',
        'sec-ch-ua-mobile': '?0',
        'sec-ch-ua-platform': '"Windows"',
    }

    today = datetime.today().date()
    # Get Weekly Thursdays if index Option
    if ticker == 'XJO':
        # Find the next Thursday
        days_until_next_thursday = (3 - today.weekday()) % 7
        next_thursday = today + timedelta(days=days_until_next_thursday)
        # Get the next Thursdays for next 52 weeks
        next_thursdays = [next_thursday + timedelta(days=7*i) for i in range(52)]
        expiry_dates = [d.strftime('%Y-%m-%d') for d in next_thursdays]
    else:
        # Find the 3rd Thursday of each month for next 6 months
        expiry_dates = []
        for i in range(6):
            year = today.year
            month = today.month + i
            if month > 12:
                year += 1
                month -= 12
            # calculate the date of the third Thursday of the month
            first_day = datetime(year, month, 1)
            offset = (3 - first_day.weekday()) % 7 + 14
            d = first_day + timedelta(days=offset)
            expiry_dates.append(str(d.date()))
    
    base_url = f'https://asx.api.markitdigital.com/asx-research/1.0/derivatives/equity/{ticker}/options/expiry-groups'

    params = {
        'callsPuts': 'all',
        'expiryDates': expiry_dates,
        'showOpenInterestOnly': 'false',
        'showTheoreticalPricesOnly': 'false',
        'styles': 'all',
        'includeLepoToressOptions': 'false'
    }

    r = requests.get(base_url,headers=headers, params=params)
    return r.json()






# Create iterator of option chain using Dataclass
def transform_data(data):
    option_data_list = []
    for item in data['data']['items']:
        expiration_date = item['date']

        for group in item['exerciseGroups']:
            call_data = {
                'expiration_date': datetime.fromisoformat(expiration_date),
                'price_exercise': group['priceExercise'], 
                'option_type': 'Call', 
                'symbol': group['call']['symbol'], 
                'open_interest': group['call'].get('openInterest'), 
                'volume': group['call'].get('volume'), 
                'price_theoretical': float(group['call'].get('priceTheoretical')), 
                'price_bid': float(group['call'].get('priceBid')), 
                'price_ask': float(group['call'].get('priceAsk'))
    
            }
            option_data_list.append(call_data)

            put_data = {
                'expiration_date': datetime.fromisoformat(expiration_date),
                'price_exercise': group['priceExercise'], 
                'option_type': 'Put', 
                'symbol': group['put']['symbol'], 
                'open_interest': group['put'].get('openInterest'), 
                'volume': group['put'].get('volume'), 
                'price_theoretical': float(group['put'].get('priceTheoretical')), 
                'price_bid': float(group['put'].get('priceBid')), 
                'price_ask': float(group['put'].get('priceAsk'))
        
            }
            option_data_list.append(put_data)

    return option_data_list

# main func
def get_option_chain(ticker):
    try:
        data = get_data(ticker)
        return pd.DataFrame(transform_data(data))
    except Exception as e:
        print(e)
        pass



# Scrape Yahoo Finance mutual fund holders table
def get_holders(ticker):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}

    url = f'https://au.finance.yahoo.com/quote/{ticker}/holders?p={ticker}'
    s = HTMLSession()
    r = s.get(url, headers=headers)
    html = HTMLParser(r.text)

    if len(html.css('table')) == 2:
        table1 = html.css('table')[1]
        header_text = [i.text().strip() for i in table1.css('th')]
        holdings_data = [[c.text().strip() for c in row.css('td')] for row in table1.css('tr')[1:]]
        result = [dict(zip(header_text, t)) for t in holdings_data]
        return pd.DataFrame(result)


    elif len(html.css('table')) == 3:
        table1 = html.css('table')[1]
        header_text = [i.text().strip() for i in table1.css('th')]
        holdings_data = [[c.text().strip() for c in row.css('td')] for row in table1.css('tr')[1:]]
        result1 = [dict(zip(header_text, t)) for t in holdings_data]
        
        table2 = html.css('table')[2]
        header_text = [i.text().strip() for i in table2.css('th')]
        holdings_data = [[c.text().strip() for c in row.css('td')] for row in table2.css('tr')[1:]]
        result2 = [dict(zip(header_text, t)) for t in holdings_data]

        return pd.DataFrame(result1 + result2)




# # Scrape earnings calender
def get_earnings(ticker):
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
    url = f'https://www.tipranks.com/stocks/au:{ticker}/earnings'
    s = HTMLSession()
    r = s.get(url, headers=headers)
    html = HTMLParser(r.text)

    earnings_data = []
    
    tbody = html.css_first('tbody.rt-tbody')
    rows = tbody.css('tr')

    for row in rows:
        forecast, eps = row.css('td')[2].text().split('/')
        forecast = forecast.strip().replace('-', '').replace('>', '').replace('<', '') or None
        eps = eps.strip().replace('-', '').replace('>', '').replace('<', '') or None
        earnings_dict = {
            'report_date': row.css('td')[0].text(),
            'forecast': forecast,
            'eps': eps
        }
        earnings_data.append(earnings_dict)

    return pd.DataFrame(earnings_data)







def get_company_directory():
    tickers = get_listed_options()
    url = 'https://asx.api.markitdigital.com/asx-research/1.0/companies/directory/file?access_token=83ff96335c2d45a094df02a206a39ff4'
    df = pd.read_csv(url)
    df = df[df['ASX code'].isin(tickers)].reset_index(drop=True)
    df.columns = df.columns.str.lower().str.replace(' ','_')
    df.listing_date = pd.to_datetime(df.listing_date)
    return df


# def get_stock_info(ticker):
#     @dataclass
#     class StockInfo:
#         sector: Optional[str] = None
#         industry: Optional[str] = None
#         marketcap: Optional[float] = None
#         enterprisevalue: Optional[float] = None
#         forwardpe: Optional[float] = None
#         ev_ebita: Optional[float] = None


#     headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/110.0.0.0 Safari/537.36'}
#     profile_url = f"https://au.finance.yahoo.com/quote/{ticker}/profile?p={ticker}"
#     s = HTMLSession()
#     r = s.get(profile_url, headers=headers)
#     html = HTMLParser(r.text)
#     # Extract
#     table = html.css_first('div.asset-profile-container')
#     data = table.css('span')
#     sector = data[1].text()
#     industry = data[3].text()

#     stats_url = f"https://au.finance.yahoo.com/quote/{ticker}/key-statistics?p={ticker}"
#     s = HTMLSession()
#     r = s.get(stats_url, headers=headers)
#     html = HTMLParser(r.text)
#     # Extract
#     table1 = html.css_first('tbody')
#     data = table1.css('tr')

#     data_dict = {}
#     data_dict['sector'] = sector
#     data_dict['industry'] = industry
#     data_dict['marketcap'] = data[0].css('td')[1].text()
#     data_dict['enterprisevalue'] = data[1].css('td')[1].text()
#     data_dict['forwardpe'] = data[3].css('td')[1].text()
#     data_dict['ev_ebita'] = data[8].css('td')[1].text()


#     return data_dict

# result = []
# total = len(tickers)
# count = 0
# for ticker in tickers:
#     try:
#         ticker_ax = ticker+'.AX'
#         data = get_stock_info(ticker_ax)
#         data['Ticker'] = ticker
#         result.append(data)
#         print(f'processing {ticker}, {count}/{total}')
#         print(data)
#         count += 1
#     except:
#         pass

# df = pd.DataFrame(result)



