import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle
from tabs import tab_1_cust, tab_2_cust, tab_3_cust, tab_4_cust
from utils_cust import display_eval_metrics, Viridis


df=pd.read_csv('resources/final_probs_cust.csv')


## Instantiante Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions'] = True
app.title='Customer Predictions!'


## Layout
app.layout = html.Div([
    html.H1('Predicting Customer Responses'),
    dcc.Tabs(id="tabs-template", value='tab-1-template', children=[
        dcc.Tab(label='Introduction', value='tab-1-template'),
        dcc.Tab(label='Model Evaluation', value='tab-2-template'),
        dcc.Tab(label='Testing Results', value='tab-3-template'),
        dcc.Tab(label='User Inputs', value='tab-4-template'),
    ]),
    html.Div(id='tabs-content-template')
])


############ Callbacks

@app.callback(Output('tabs-content-template', 'children'),
              [Input('tabs-template', 'value')])
def render_content(tab):
    if tab == 'tab-1-template':
        return tab_1_cust.tab_1_layout
    elif tab == 'tab-2-template':
        return tab_2_cust.tab_2_layout
    elif tab == 'tab-3-template':
        return tab_3_cust.tab_3_layout
    elif tab == 'tab-4-template':
        return tab_4_cust.tab_4_layout

# Tab 2 callbacks

@app.callback(Output('page-2-graphic', 'figure'),
              [Input('page-2-radios', 'value')])
def radio_results(value):
    return display_eval_metrics(value)

# Tab 3 callback # 1
@app.callback(Output('page-3-content', 'children'),
              [Input('page-3-dropdown', 'value')])
def page_3_dropdown(value):
    id=df.loc[value, 'ID']
    return f'You have selected "{id}"'

# Tab 3 callback # 2
@app.callback(Output('response-prob', 'children'),
              [Input('page-3-dropdown', 'value')])
def page_3_survival(value):
    response=df.loc[value, 'response_prob']
    actual=df.loc[value, 'Responded']
    responded=round(response*100)
    return f'Predicted probability of responding is {response}%, Actual response is {actual}'

# Tab 3 callback # 2
@app.callback(Output('response-characteristics', 'children'),
              [Input('page-3-dropdown', 'value')])
def page_3_characteristics(value):
    mydata=df.drop(['Responded', 'response_prob', 'ID'], axis=1)
    mydata=df[['PriorCampaign', 'Children',
                'Age', 'Single', 'NewlySingle', 'Relationship',
                'UnderGraduate', 'Graduate', 'PostGraduate',
                'Income (1, 25000]', 'Income (25000, 50000]', 'Income (50000, 75000]', 'Income (75000, 100000]', 'Income (100000, 670000]',
                'Recency (0, 25]', 'Recency (25, 50]', 'Recency (50, 75]', 'Recency (75, 100]',
                'MntSpent (0, 600]', 'MntSpent (600, 1200]', 'MntSpent (1200, 1800]', 'MntSpent (1800, 2400]', 'MntSpent (2400, 3000]',
                'WebVisitsMoreThan10', 'CustYrsMoreThan7', 'NumPurchasesMoreThan25']]
    return html.Table(
        [html.Tr([html.Th(col) for col in mydata.columns])] +
        [html.Tr([
            html.Td(mydata.iloc[value][col]) for col in mydata.columns
        ])]
    )

# Tab 4 Callback # 1
@app.callback(Output('user-inputs-box', 'children'),
            [
              Input('Education_dropdown', 'value'),
              Input('Age_dropdown', 'value'),
              Input('Income_dropdown', 'value'),
              Input('MaritalStatus_dropdown', 'value'),
              Input('Spending_dropdown', 'value'),
              Input('NumPurchases_dropdown', 'value'),
              Input('Recency_dropdown', 'value'),
              Input('Webvisits_dropdown', 'value'),
              Input('children_radio', 'value'),
              Input('priorCmpgn_radio', 'value'),
              Input('customerSince_radio', 'value')
              ])
def update_user_table(edu, age, income, mstatus, spending, nump, recency, visits, children, priorCmpgn, customerSince):
    return html.Div([
        html.Div(f'Education: {edu}'),
        html.Div(f'Age: {age}'),
        html.Div(f'Income: {income}'),
        html.Div(f'Marital Status: {mstatus}'),
        html.Div(f'Children in household: {children}'),
        html.Div(f'Customer since (in years): {customerSince}'),
        html.Div(f'Amount Spent: {spending}'),
        html.Div(f'Number of Purchases: {nump}'),
        html.Div(f'Last Purchased (days ago): {recency} '),
        html.Div(f'Number of web visits: {visits}'),
        html.Div(f'Responded to prior campaigns: {priorCmpgn}'),
    ])

# Tab 4 Callback # 2
@app.callback(Output('final_prediction', 'children'),
            [
              Input('Education_dropdown', 'value'),
              Input('Age_dropdown', 'value'),
              Input('Income_dropdown', 'value'),
              Input('MaritalStatus_dropdown', 'value'),
              Input('Spending_dropdown', 'value'),
              Input('NumPurchases_dropdown', 'value'),
              Input('Recency_dropdown', 'value'),
              Input('Webvisits_dropdown', 'value'),
              Input('children_radio', 'value'),
              Input('priorCmpgn_radio', 'value'),
              Input('customerSince_radio', 'value')
              ])
def final_prediction(edu, age, income, mstatus, spending, nump, recency, visits, children, priorCmpgn, customerSince):
    inputs=[mstatus, priorCmpgn, children, age, edu, income, recency, spending, visits, customerSince, nump]
    keys=['Marital_Status', 'PriorCampaign', 'Children', 'age', 'Education', 'Income', 'Recency', 'MntSpent', 'NumWebVisitsMonth', 'CustYrsMoreThan7', 'NumPurchases']
    dict11=dict(zip(keys, inputs))
    df=pd.DataFrame([dict11])
    # create the features we'll need to run our logreg model.
    df['age']=pd.to_numeric(df.age, errors='coerce')
    df['Single']=np.where((df.Marital_Status=='Single'),1,0)
    df['Relationship']=np.where((df.Marital_Status=='Relationship'),1,0)
    df['NewlySingle']=np.where((df.Marital_Status=='NewlySingle'),1,0)

    df['Children']=np.where((df.Children=='Yes'),1,0)
    df['PriorCampaign']=np.where((df.PriorCampaign=='Yes'),1,0)
    df['Income']=pd.to_numeric(df.Income, errors='coerce')
    df['Recency']=pd.to_numeric(df.Recency, errors='coerce')
    df['MntSpent']=pd.to_numeric(df.MntSpent, errors='coerce')
    df['NumWebVisitsMonth']=pd.to_numeric(df.NumWebVisitsMonth, errors='coerce')
    df['CustYrsMoreThan7']=np.where((df.CustYrsMoreThan7=='Yes'),1,0)
    df['NumPurchases']=pd.to_numeric(df.NumPurchases, errors='coerce')

    df['Graduate']=np.where(df.Education=='Graduate',1,0)
    df['PostGraduate']=np.where(df.Education=='PostGraduate',1,0)
    df['UnderGraduate']=np.where(df.Education=='UnderGraduate',1,0)

    df['Age']=np.where((df.age>50),1,0)

    df['Income125000']=np.where((df.Income>=1)&(df.Income<25000),1,0)
    df['Income2500050000']=np.where((df.Income>=25000)&(df.Income<50000),1,0)
    df['Income5000075000']=np.where((df.Income>=50000)&(df.Income<75000),1,0)
    df['Income75000100000']=np.where((df.Income>=75000)&(df.Income<100000),1,0)
    df['Income100000670000']=np.where((df.Income>=100000)&(df.Income<670000),1,0)

    df['Recency025']=np.where((df.Recency>=0)&(df.Recency<25),1,0)
    df['Recency2550']=np.where((df.Recency>=25)&(df.Recency<50),1,0)
    df['Recency5075']=np.where((df.Recency>=50)&(df.Recency<75),1,0)
    df['Recency75100']=np.where((df.Recency>=75)&(df.Recency<100),1,0)

    df['MntSpent0600']=np.where((df.MntSpent>=0)&(df.MntSpent<600),1,0)
    df['MntSpent6001200']=np.where((df.MntSpent>=600)&(df.MntSpent<1200),1,0)
    df['MntSpent12001800']=np.where((df.MntSpent>=1200)&(df.MntSpent<1800),1,0)
    df['MntSpent18002400']=np.where((df.MntSpent>=1800)&(df.MntSpent<2400),1,0)
    df['MntSpent24003000']=np.where((df.MntSpent>=2400)&(df.MntSpent<3000),1,0)

    df['WebVisitsMoreThan10']=np.where((df.NumWebVisitsMonth>10),1,0)

    df['NumPurchasesMoreThan25']=np.where((df.NumPurchases>25),1,0)

    # drop unnecessary columns, and reorder columns to match the logreg model.
    df=df.drop(['age', 'Marital_Status', 'Income', 'Education', 'Recency', 'MntSpent', 'NumWebVisitsMonth', 'NumPurchases'], axis=1)
    df=df[['PriorCampaign', 'Children', 'Age',
           'Single', 'NewlySingle', 'Relationship',
           'UnderGraduate', 'Graduate', 'PostGraduate',
           'Income125000', 'Income2500050000','Income5000075000', 'Income75000100000', 'Income100000670000',
           'Recency025', 'Recency2550', 'Recency5075', 'Recency75100',
           'MntSpent0600', 'MntSpent6001200', 'MntSpent12001800', 'MntSpent18002400','MntSpent24003000',
           'WebVisitsMoreThan10', 'CustYrsMoreThan7', 'NumPurchasesMoreThan25']]

    # unpickle the final model
    file = open('resources/final_logreg_model_cust.pkl', 'rb')
    logreg=pickle.load(file)
    file.close()
    # predict on the user-input values (need to create an array for this)
    firstrow=df.loc[0]
    print('firstrow', firstrow)
    myarray=firstrow.values
    print('myarray', myarray)
    thisarray=myarray.reshape((1, myarray.shape[0]))
    print('thisarray', thisarray)

    prob=logreg.predict_proba(thisarray)
    final_prob=round(float(prob[0][1])*100,1)
    return(f'Probability of Responding: {final_prob}%')


####### Run the app #######
if __name__ == '__main__':
    app.run_server(debug=True)
