import dash
from dash.dependencies import Input, Output, State
from dash import dcc, html
import plotly.graph_objs as go
import pandas as pd
import numpy as np
import pickle
from tabs import tab_1, tab_2, tab_3, tab_4
from utils import display_eval_metrics, Viridis


df=pd.read_csv('resources/final_probs.csv')


## Instantiante Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.config['suppress_callback_exceptions'] = True
app.title='Titanic!'


## Layout
app.layout = html.Div([
    html.H1('Surviving the Titanic'),
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
        return tab_1.tab_1_layout
    elif tab == 'tab-2-template':
        return tab_2.tab_2_layout
    elif tab == 'tab-3-template':
        return tab_3.tab_3_layout
    elif tab == 'tab-4-template':
        return tab_4.tab_4_layout

# Tab 2 callbacks

@app.callback(Output('page-2-graphic', 'figure'),
              [Input('page-2-radios', 'value')])
def radio_results(value):
    return display_eval_metrics(value)

# Tab 3 callback # 1
@app.callback(Output('page-3-content', 'children'),
              [Input('page-3-dropdown', 'value')])
def page_3_dropdown(value):
    name=df.loc[value, 'Name']
    return f'You have selected "{name}"'

# Tab 3 callback # 2
@app.callback(Output('survival-prob', 'children'),
              [Input('page-3-dropdown', 'value')])
def page_3_survival(value):
    survival=df.loc[value, 'survival_prob']
    actual=df.loc[value, 'Survived']
    survival=round(survival*100)
    return f'Predicted probability of survival is {survival}%, Actual survival is {actual}'

# Tab 3 callback # 2
@app.callback(Output('survival-characteristics', 'children'),
              [Input('page-3-dropdown', 'value')])
def page_3_characteristics(value):
    mydata=df.drop(['Survived', 'survival_prob', 'Name'], axis=1)
    mydata=df[['Siblings and Spouses',
                 'female',
                 'Cabin Class 2',
                 'Cabin Class 3',
                 'Cherbourg',
                 'Queenstown',
                 'Age (20, 28]',
                 'Age (28, 38]',
                 'Age (38, 80]',
                 'Mrs.',
                 'Miss',
                 'VIP']]
    return html.Table(
        [html.Tr([html.Th(col) for col in mydata.columns])] +
        [html.Tr([
            html.Td(mydata.iloc[value][col]) for col in mydata.columns
        ])]
    )

# Tab 4 Callback # 1
@app.callback(Output('user-inputs-box', 'children'),
            [
              Input('family_dropdown', 'value'),
              Input('age_dropdown', 'value'),
              Input('cabin_dropdown', 'value'),
              Input('title_radio', 'value'),
              Input('sex_radio', 'value'),
              Input('port_radio', 'value')
              ])
def update_user_table(family, age, cabin, title, sex, embark):
    return html.Div([
        html.Div(f'Family Members: {family}'),
        html.Div(f'Age: {age}'),
        html.Div(f'Cabin Class: {cabin}'),
        html.Div(f'Title: {title}'),
        html.Div(f'Sex: {sex}'),
        html.Div(f'Embarkation: {embark}'),
    ])

# Tab 4 Callback # 2
@app.callback(Output('final_prediction', 'children'),
            [
              Input('family_dropdown', 'value'),
              Input('age_dropdown', 'value'),
              Input('cabin_dropdown', 'value'),
              Input('title_radio', 'value'),
              Input('sex_radio', 'value'),
              Input('port_radio', 'value')
              ])
def final_prediction(family, age, cabin, title, sex, embark):
    inputs=[family, age, cabin, title, sex, embark]
    keys=['family', 'age', 'cabin', 'title', 'sex', 'embark']
    dict6=dict(zip(keys, inputs))
    df=pd.DataFrame([dict6])
    # create the features we'll need to run our logreg model.
    df['age']=pd.to_numeric(df.age, errors='coerce')
    df['family']=pd.to_numeric(df.family, errors='coerce')
    df['third']=np.where(df.cabin=='Third',1,0)
    df['second']=np.where(df.cabin=='Second',1,0)
    df['female']=np.where(df.sex=='Female',1,0)
    df['cherbourg']=np.where(df.embark=='Cherbourg',1,0)
    df['queenstown']=np.where(df.embark=='Queenstown',1,0)
    df['age2028']=np.where((df.age>=20)&(df.age<28),1,0)
    df['age2838']=np.where((df.age>=28)&(df.age<38),1,0)
    df['age3880']=np.where((df.age>=38)&(df.age<80),1,0)
    df['mrs']=np.where(df.title=='Mrs.', 1,0)
    df['miss']=np.where(df.title=='Miss', 1,0)
    df['vip']=np.where(df.title=='VIP', 1,0)
    # drop unnecessary columns, and reorder columns to match the logreg model.
    df=df.drop(['age', 'cabin', 'title', 'sex', 'embark'], axis=1)
    df=df[['family', 'female', 'second', 'third', 'cherbourg', 'queenstown', 'age2028',
    'age2838', 'age3880', 'mrs', 'miss', 'vip']]
    # unpickle the final model
    file = open('resources/final_logreg_model.pkl', 'rb')
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
    return(f'Probability of Survival: {final_prob}%')



####### Run the app #######
if __name__ == '__main__':
    app.run_server(debug=True)
