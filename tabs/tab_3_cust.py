import dash
from dash import dcc, html
import pandas as pd


filepath='resources/final_probs_cust.csv'
df=pd.read_csv(filepath)
ids=df['ID'].values
index=df['ID'].index.values
idslist = list(zip(index, ids))

tab_3_layout = html.Div([
    html.H3('Results for Testing Dataset'),
    html.Div([
        html.Div([
            html.Div('Select a dataset id to view their predicted response:'),
            dcc.Dropdown(
                id='page-3-dropdown',
                options=[{'label': k, 'value': i} for i,k in idslist],
                value=idslist[0][0]
            ),

        ],className='one column'),
        html.Div([
            html.Div(id='page-3-content', style={'fontSize':18}),
            html.Div(id='response-prob', style={'fontSize':18, 'color':'red'}),
            html.Table(id='response-characteristics')
        ],className='nine columns'),
    ],className='twelve columns'),

])
