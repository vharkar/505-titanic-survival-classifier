import dash
from dash import dcc, html
import pandas as pd

filepath='resources/final_probs_cust.csv'
df=pd.read_csv(filepath)
names=df['ID'].values
index=df['ID'].index.values
idList = list(zip(index, names))

tab_4_layout = html.Div([
    html.H3('How does a customer respond to a campaign?'),
    html.Div([
        html.Div('Select:', className='one column'),
        # Title,
        html.Div([
            html.Div('Education'),
            dcc.Dropdown(
                id='Education_dropdown',
                options=[{'label': i, 'value': i} for i in ['UnderGraduate', 'Graduate', 'PostGraduate']],
                value='UnderGraduate',
                ),
        ],className='three columns'),
        html.Div([
            html.Div('Age'),
            dcc.Dropdown(
                id='Age_dropdown',
                options=[{'label': i, 'value': i} for i in range(20,80)],
                value='25',
                ),
        ],className='three columns'),
        html.Div([
            html.Div('Income'),
            dcc.Dropdown(
                id='Income_dropdown',
                options=[{'label': i, 'value': i} for i in range(1,670000)],
                value='75000',
                ),
        ],className='four columns'),

        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        html.Div([
            html.Div('Spending'),
            dcc.Dropdown(
                id='Spending_dropdown',
                options=[{'label': i, 'value': i} for i in range(0, 2600)],
                value='1200',
            ),
        ], className='four columns'),
        html.Div([
            html.Div('Number of Purchases in the past month'),
            dcc.Dropdown(
                id='NumPurchases_dropdown',
                options=[{'label': i, 'value': i} for i in range(0, 50)],
                value='5',
            ),
        ], className='four columns'),

        html.Br(),
        html.Br(),
        html.Br(),
        html.Br(),

        html.Div([
            html.Div('Days since most recent purchase'),
            dcc.Dropdown(
                id='Recency_dropdown',
                options=[{'label': i, 'value': i} for i in range(0, 100)],
                value='30',
            ),
        ], className='four columns'),
        html.Div([
            html.Div('Number of Web visits per month'),
            dcc.Dropdown(
                id='Webvisits_dropdown',
                options=[{'label': i, 'value': i} for i in range(0, 20)],
                value='5',
            ),
        ], className='four columns'),
        html.Div('     ', className='one column')
    ],className='twelve columns'),

    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),

    html.Div([
        html.Div('Select:', className='one column'),
        html.Div([
            html.Div('Marital Status'),
            dcc.RadioItems(
                id='MaritalStatus_dropdown',
                options=[{'label': i, 'value': i} for i in ['Single', 'NewlySingle', 'Relationship']],
                value='Single',
                ),
        ],className='three columns'),
        html.Div([
            html.Div('Children in Household'),
            dcc.RadioItems(
                id='children_radio',
                options=[{'label': i, 'value': i} for i in ['Yes', 'No']],
                value='None',
                ),
        ],className='three columns'),
        html.Div([
            html.Div('Responded to Prior Campaign'),
            dcc.RadioItems(
                id='priorCmpgn_radio',
                options=[{'label': i, 'value': i} for i in ['Yes', 'No']],
                value='None',
                ),
        ],className='five columns'),
        html.Div([
            html.Div('Customer for more than 7 years'),
            dcc.RadioItems(
                id='customerSince_radio',
                options=[{'label': i, 'value': i} for i in ['Yes', 'No']],
                value='None',
            ),
        ], className='five columns'),
    ],className='twelve columns'),
    html.Br(),
    html.Br(),
    html.Br(),
    html.Br(),
    # Output results
    html.Div([
        html.Div(id='user-inputs-box', style={'text-align':'center','fontSize':18}),
        html.Div(id='final_prediction', style={'color':'red','text-align':'center','fontSize':18})
    ],className='twelve columns'),



])
