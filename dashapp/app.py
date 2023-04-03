# Import libraries
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, Input, Output, ctx, dash_table
import plotly.express as px
import plotly.graph_objs as go
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
from datetime import datetime, timedelta
import yfinance as yf
from google.cloud import bigquery


config = {
    'displayModeBar': False,
    'displaylogo': False,                                       
    'modeBarButtonsToRemove': ['zoom2d', 'hoverCompareCartesian', 'hoverClosestCartesian', 'toggleSpikelines']
}

######################## Back-end #############################
# import os
# os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'bqtest-379921-c8ee254b4e40.json'
# bq_client=bigquery.Client()

# get list of tickers
sql = f"""
    SELECT DISTINCT(ticker)
    FROM `bqtest-379921.asxdata.options`
"""
tickers_list = bq_client.query(sql)
tickers_list = sorted([i[0] for i in tickers_list])
ticker_options = [{'label': ticker, 'value': ticker} for ticker in tickers_list]

# Get tables
def get_directory():
    sql = f"""SELECT asx_code, company_name, gics_industry_group, market_cap
      FROM `bqtest-379921.asxdata.companydirectory`;
      """
    df = bq_client.query(sql).to_dataframe()
    return df
directory_df = get_directory()

# query optionchains
def query_optionchain():
    sql = f"""
        SELECT DATE(expiration_date) AS expiration_date,
        price_exercise, option_type, open_interest, volume, price_theoretical, ticker
        FROM `bqtest-379921.asxdata.options`;
    """
    df = bq_client.query(sql).to_dataframe()
    return df

optionchain = query_optionchain()
optionchain = optionchain.query('ticker != "STW"')


# XJO Index data for surface plot
def surface_plot_data():
    # query optionchains
    sql = f"""
        SELECT DATE(expiration_date) AS expiration_date,
        price_exercise, option_type, open_interest, volume, price_theoretical, ticker
        FROM `bqtest-379921.asxdata.options`
        WHERE ticker = 'XJO';
    """
    df = bq_client.query(sql).to_dataframe()

    S = yf.download("^AXJO", start= datetime.today().date() - timedelta(days=30))['Close'][-1]

    # define Black-Scholes formula
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

    # calculate Delta, Gamma, and IV for each option in the dataframe
    # add delta column to dataframe
    df['delta'] = df.apply(lambda row: bs_delta(S, row['price_exercise'], 0, 0.2, (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365, row['option_type']), axis=1)
    # add gamma column to dataframe
    df['gamma'] = df.apply(lambda row: bs_gamma(S, row['price_exercise'], 0, 0.2, (pd.to_datetime(row['expiration_date']) - pd.to_datetime('today')).days / 365), axis=1)
    # add implied volatility column to dataframe
    df['implied_volatility'] = df.apply(lambda row: implied_volatility(row, S), axis=1)
    df['implied_volatility'] = round(df['implied_volatility'],3)
    df = df.fillna(0)
    df = df.sort_values(by='price_exercise')
    return df

df = surface_plot_data()
today = datetime.today().date()
df['dte'] = [(date - today).days for date in df.expiration_date.tolist()]

def create_surface_plot(data, label='implied_volatility'):
    # Create a mesh grid
    expiration_dates = sorted(data['dte'].unique())
    price_exercises = sorted(data['price_exercise'].unique())
    X, Y = np.meshgrid(price_exercises, expiration_dates, )
    Z = np.zeros_like(X)

    for i, row in data.iterrows():
        y_idx = expiration_dates.index(row['dte'])
        x_idx = price_exercises.index(row['price_exercise'])
        Z[y_idx][x_idx] = row[label]

    fig = go.Figure(go.Surface(
        contours = {
        "x": {"show": True, "start": 1.5, "end": 2, "size": 1, "color":"white"},
        "z": {"show": True, "start": 0.5, "end": 0.8, "size": 0.05}
        },
        x=X, y=Y, z=Z, showscale=False,
        hovertemplate='Strike: %{x}<br>Days to expiration: %{y}<br>Implied volatility: %{z}<extra></extra>'))
    
    fig.update_layout(
        margin=dict(l=10, r=10, b=10, t=10),
        width=500, height=400, template='none', 
        scene=dict(
            bgcolor='white',
            xaxis=dict(
                tickfont=dict(size=10),
                title='Strike',
                title_font=dict(size=10)
            ),
            yaxis=dict(
                tickfont=dict(size=10),
                title='Days to expiration',
                title_font=dict(size=10)
            ),
            zaxis=dict(
                tickfont=dict(size=10),
                title='Implied volatility',
                title_font=dict(size=10)
            ),
            camera=dict(eye=dict(x=1.8, y=1.8, z=.8)
            )))

    return fig

vol_surface = create_surface_plot(df)




def get_price_details(ticker):
    if ticker == 'XJO':
        ticker = '^AXJO'
    else:
        ticker = ticker + '.AX'
    prices =  yf.download(ticker, start= datetime.today().date() - timedelta(days=5))
    last_price = prices['Close'][-1]
    change = last_price - prices['Close'][-2]
    percent_change = change / prices['Close'][-2] * 100
    is_positive = change >= 0

    price_str = f"{last_price:.2f}"
    change_str = f"{change:+.2f} ({percent_change:+.2f}%)"
    change_color = "green" if is_positive else "red"

    time_str = prices.index.max()
    time_str = time_str.strftime("%B %d, %Y")
    date_str = f"(AS OF {time_str})"
    return last_price, price_str, change_str, change_color, date_str


# Option DataTable
agg_df = optionchain.groupby(['ticker', 'option_type'])[['open_interest', 'volume']].sum().reset_index()
agg_df = agg_df.pivot_table(index='ticker', columns='option_type', values=['open_interest', 'volume'], aggfunc='sum')
agg_df.columns = ['_'.join(col).lower() for col in agg_df.columns]
agg_df.reset_index(inplace=True)
agg_df['put/call_oi'] = agg_df.open_interest_put / agg_df.open_interest_call
df = pd.merge(agg_df, directory_df[['asx_code','gics_industry_group', 'market_cap']], how='left', left_on='ticker', right_on='asx_code')

df.loc[df['ticker']=='XJO', 'gics_industry_group'] = 'Index'
df.drop(columns=['asx_code'], inplace=True)
df.fillna('0', inplace=True)
df.market_cap = df.market_cap.astype(int)

df = df[[
    'ticker', 
    'gics_industry_group', 
    'market_cap', 
    'open_interest_call', 
    'open_interest_put',
    'put/call_oi',
    ]]
column_names = ['Ticker', 'Sector', 'Market Cap', 'Call OI', 'Put OI', 'P/C OI']
df.columns = column_names
df = round(df,2)










################ Front end #####################
app = dash.Dash(external_stylesheets=[dbc.themes.ZEPHYR])


#Layout 
app.layout = dbc.Container(
    [
        dbc.Row(
            [
                dbc.Col(html.H3('optionview',style={"font-size": "18px", 'font-weight':'bold'}))
            ], 
            style={'height': '5vh', 'margin': '10px'}),

        dbc.Row(html.Div(
                [
                    html.H1(id="ticker", style={'text-align': 'start', "font-size": "26px", "font-weight": "bold"}),
                    html.H2(id="name", style={'text-align': 'start', 'font-size':'14px'}),
                    html.H3(id="sector-details", style={'text-align': 'start', 'font-size':'14px'})
                ]), justify='start', style={"height": "10vh", "margin": "10px"}),


        dbc.Row([
            dbc.Col(html.Div(id="price_str", style={"font-size": "26px", 'font-weight': 'bold'}), width="auto", align='end'),
            dbc.Col(html.Div([
                html.Div(id="change_str", style={"font-size": "12px"}),
            ], style={"height":"2.5vh", "margin":'10px'}), width="auto", align='end'),
        ], style={"height": "5vh", "margin-left": "10px"}),
        
        dbc.Row([html.Div(id="date_str", style={"font-size": "10px", "text-align": "left"})],
                style={"height": "1vh", "margin-left": "10px"}),


        dbc.Row(
            [
                dbc.Col(dcc.Dropdown(id='ticker-dropdown',
                                     options=ticker_options,
                                     value='XJO'), width=2),

                dbc.Col(dcc.Dropdown(id='expiration-dropdown',
                                     options=[],
                                     value=None), width=3),
            ], justify='start',
            style={'height': '5vh', 'margin': '10px'}),


        dbc.Row(
            [
                dbc.Col(
                    [
                        dbc.Button('Open Interest', outline=True, color="secondary", id='open-interest-button', n_clicks=0, value='open_interest', size='sm', className="me-md-2"),
                        dbc.Button('Volume', outline=True, color="secondary", id='volume-button', n_clicks=0, size='sm', value='volume', className="me-md-2"),
                        dbc.Button('Probability Distribution', outline=True, color="secondary", id='probability-button', size='sm', value='prob', className="me-md-2"),
                        dbc.Button('Volatility Skew', outline=True, color="secondary", id='volatility-button', size='sm', className="me-md-2"),
                        dbc.Button('Delta', outline=True, color="secondary", id='delta-button', size='sm', className="me-md-2"),
                        dbc.Button('Gamma', outline=True, color="secondary", id='gamma-button', size='sm', className="me-md-2"),
                        dbc.Button('Theta', outline=True, color="secondary", id='theta-button', size='sm', className="me-md-2"),
                        dbc.Button('Vega', outline=True, color='secondary', size='sm',className="me-md-2"),
                    ],
                    width='auto'
                )
            ],
            justify='start', style={'height': '5vh', 'margin': '10px'}
        ),


        dbc.Row(
            [
                dcc.Store(id='optionchain-data'),
                dcc.Store(id='last-price'),
                html.H4(id="metric-title", className='font-weight-bold', style={"text-align": "center", "font-size":"18px", "z-index":'1'}),
                html.H5(id="graph-subtitle", style={"text-align": "center", "font-size":"12px", 'z-index':'1'}),
                dbc.Col(dbc.Spinner(dcc.Graph(id='graph', config=config))),
            ], 
            style={'height': '50vh',
                   'margin': '10px'}),


        dbc.Row(
            [
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H1(id="call-metric", className="card-title",style={"font-size": "14px", 'text-align': 'center',}),
                        html.P(id="subt1", className="card-subtitle", style={"font-size": "12px", 'text-align': 'center',}),
                    ]), style={"border": "none"}
                ), width='auto'),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H2(id="put-metric", className="card-title", style={"font-size": "14px", 'text-align': 'center',}),
                        html.P(id="subt2", className="card-subtitle", style={"font-size": "12px", 'text-align': 'center',}),
                    ]), style={"border": "none"}
                ), width='auto'),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H3(id="metric-total", className="card-title", style={"font-size": "14px", 'text-align': 'center',}),
                        html.P(id="subt3", className="card-subtitle", style={"font-size": "12px", 'text-align': 'center',}),
                    ]), style={"border": "none"}
                ), width='auto'),
                dbc.Col(dbc.Card(
                    dbc.CardBody([
                        html.H4(id="putcall-metric", className="card-title", style={"font-size": "14px", 'text-align': 'center',}),
                        html.P(id="subt4", className="card-subtitle", style={"font-size": "12px", 'text-align': 'center',}),
                    ]), style={"border": "none"}
                ), width='auto'),
            ], 
            justify='center', style={"height": "10vh", "margin":"10px"}),

        dbc.Row(
            [
                dbc.Col(
                    [
                        html.Div(
                            [
                                html.H4("ASX Option DataTable", className='font-weight-bold', style={"text-align": "center", "font-size":"18px"}),
                                html.H5("(End-of-day data only)", style={"text-align": "center", "font-size": "12px", "z-index": "1"}),
                                dash_table.DataTable(
                                    data=df.to_dict('records'),
                                    columns=[{"name": i, "id": i} for i in df.columns],
                                    sort_action='native',
                                    style_table={"height":'400px', 'overflowY':'auto','overflowX':'auto', 'fontSize': '12px'},
                                    fixed_rows={"headers": True},
                                    editable=True,
                                    style_cell={'textAlign': 'center'},
                                    style_header={
                                        'fontWeight': 'bold', 'textAlign':'center'
                                    },
                                    style_data_conditional=[
                                        {
                                            'if': {'column_id':'Ticker'},
                                            'width':'80px'
                                        }, 
                                        {
                                            'if': {'column_id':'Sector'},
                                            'width':'100px'
                                        }, 
                                        {
                                            'if': {'filter_query':'{P/C OI} > 1', 'column_id':'P/C OI'},
                                            'backgroundColor': '#EF553B',
                                            'color': 'white',
                                        }, 
                                        {
                                            'if': {'filter_query':'{P/C OI} < 1', 'column_id': 'P/C OI'},
                                            'backgroundColor': '#00CC96',
                                            'color': 'white'
                                        }, 
                                        {
                                            'if': {
                                                'state': 'active'
                                            },
                                        'backgroundColor': 'rgba(0, 116, 217, 0.3)',
                                        'border': '1px solid rgb(0, 116, 217)'
                                        },
                                    ]
                                )
                            ]   
                                )
                    ],
                    width=7,
                        ),

                dbc.Col(
                [
                    html.H4("XJO Options", className="font-weight-bold", style={"text-align": "center", "font-size": "18px", "z-index": "1"}),
                    html.H5("Implied Volatility Surface", style={"text-align": "center", "font-size": "12px", "z-index": "1"}),
                    dcc.Graph(
                        id="vol-surface",
                        figure=vol_surface,
                        config=config,
                        style={"display": "block", "margin-top": "-10px", "position": "relative", "z-index": "1"},
                        ),
                ],
                    width=5,
                    style={"display": "flex", "flex-direction": "column", "align-items": "center", "justify-content": "flex-start"},
                ),
            ],
            style={"height": "50vh", "margin": "10px"},
        ),


        dbc.Row(html.Div(
                [
                    html.H4('optionview', style={'text-align': 'center', 'font-weight': 'bold', 'font-size': '16px'}),
                    html.H5('Disclaimer: The data provided is updated at the end of the market and may be inaccurate. It should not be relied upon for making investment decisions.', style={'text-align': 'center', 
                                                        'font-weight': 'bold', 
                                                        'font-size': '12px'}),
                ]), style={'display': 'flex', 'justify-content': 'center', 'align-items': 'center', 'height': '10vh', 'margin-top':'50px'}),

    ],
)




################### Callbacks #########################
@app.callback(
    Output('ticker', 'children'),
    Output('name', 'children'),
    Output('sector-details', 'children'),
    Output('last-price', 'data'),
    Output('price_str', 'children'),
    Output('change_str', 'children'),
    Output('change_str', 'style'),
    Output('date_str', 'children'),
    Input('ticker-dropdown', 'value'),
)
def get_ticker_info(ticker):
    if ticker == 'XJO':
        last_price, price_str, change_str, change_color, date_str = get_price_details(ticker)
        return 'XJO', 'S&P/ASX 200 Index', 'Index', last_price, price_str, change_str, {"color": change_color}, date_str
    else:
        company_data = directory_df[directory_df.asx_code == ticker]
        name = company_data['company_name'].values[0]
        sector = company_data['gics_industry_group'].values[0]

        last_price, price_str, change_str, change_color, date_str = get_price_details(ticker)
        return ticker, name, sector, last_price, price_str, change_str, {"color": change_color}, date_str


@app.callback(
    Output('expiration-dropdown', 'options'),
    Input('ticker-dropdown', 'value'),
)
def update_expiration_options(ticker):
    df = optionchain[optionchain.ticker==ticker]

    options = [{'label': date, 'value':date} for date in sorted(df.expiration_date.unique())]
    return options



@app.callback(
    Output('graph', 'figure'),
    Output('metric-title', 'children'),
    Output('graph-subtitle', 'children'),
    Output('call-metric', 'children'),
    Output('put-metric', 'children'),
    Output('metric-total', 'children'),
    Output('putcall-metric', 'children'),
    Output('subt1', 'children'),
    Output('subt2', 'children'),
    Output('subt3', 'children'),
    Output('subt4', 'children'),
    Input('expiration-dropdown', 'value'),
    Input('open-interest-button', 'n_clicks'),
    Input('volume-button', 'n_clicks'),
    Input('ticker-dropdown', 'value'),
    Input('last-price', 'data'),
    prevent_initial_call=True
)
def update_figure(expiration, _, __, ticker, last_price):
    # Set default
    metric = 'open_interest'
    name = 'Open Interest'

    # Set button clicked
    button_clicked = ctx.triggered_id
    if button_clicked == 'open-interest-button':
        metric = 'open_interest'
        name = 'Open Interest'

    if button_clicked == 'volume-button':
        metric = 'volume'
        name = 'Volume'


    optionchain_filtered = optionchain[optionchain.ticker==ticker]
    if expiration is None:
        expiration = optionchain_filtered.expiration_date.min()
    else:
        expiration = datetime.strptime(expiration, "%Y-%m-%d").date()
    
    dte = (expiration - datetime.today().date()).days
    formatted_date = expiration.strftime("%B %d, %Y")
    subtitle = f'{ticker} {formatted_date} ({dte} days)'

    # Filter option chain data based on expiration date
    df = optionchain_filtered[optionchain_filtered.expiration_date == expiration]


    # Filter Puts and Call statistics
    calls = df.query('option_type == "Call"')
    puts = df.query('option_type == "Put"')
    call_metric = calls[metric].sum()
    put_metric = puts[metric].sum()
    metric_total = call_metric + put_metric
    putcall_metric_ratio = round(put_metric / call_metric,2)
    subt1 = f"Call {name} Total"
    subt2 = f"Put {name} Total"
    subt3 = f"{name} Total"
    subt4 = f"Put/Call {name} Total"
    
    call_metric = '{:,}'.format(call_metric)
    put_metric = '{:,}'.format(put_metric)
    metric_total = '{:,}'.format(metric_total)
    putcall_metric_ratio = '{:,}'.format(putcall_metric_ratio)

    fig = px.histogram(df, x='price_exercise', y=metric, color='option_type',
                        histfunc='sum', barmode='group', nbins=len(df.price_exercise.unique()),
                        labels={'option_type': 'Option Type', 
                                'price_exercise': 'Strikes', 
                                'open_interest':'Open Interest',
                                'volume': 'Volume'},
                        color_discrete_map={'Call': '#00CC96', 'Put': '#EF553B'},
                        hover_data={'price_exercise': ':.2f', metric: True},
                        template='plotly_white')

    fig.add_vline(x=last_price, line_dash='dot', line_color='black')

    # Set the layout
    fig.update_layout(
        transition_duration=500,
        width=1200,
        height=400,
        margin=dict(t=10, b=100, l=100, r=100),
        legend=dict(orientation='h', yanchor='bottom', y=-.25, xanchor='center', x=0.5,
                    title=None, font=dict(size=10)),
        xaxis=dict(fixedrange=True, linecolor='lightgray', title=dict(text='Strikes', font=dict(size=10)),
                tickfont=dict(size=10), showgrid=False),

        yaxis=dict(fixedrange=True,showline=False, title=dict(text=name, font=dict(size=10)),
                tickfont=dict(size=10), showgrid=True),
    )

    return fig,name,subtitle, call_metric,put_metric,metric_total,putcall_metric_ratio, subt1,subt2,subt3,subt4


if __name__ == '__main__':
    app.run_server(debug=False, host='0.0.0.0', port=8080)
    # app.run_server(debug=True)






