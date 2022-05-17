import pandas as pd
import plotly
import plotly.graph_objs as go
import pickle
from sklearn.metrics import roc_auc_score
from tabs.tab_2 import choices
import json

Viridis=[
"#440154", "#440558", "#450a5c", "#450e60", "#451465", "#461969",
"#461d6d", "#462372", "#472775", "#472c7a", "#46307c", "#45337d",
"#433880", "#423c81", "#404184", "#3f4686", "#3d4a88", "#3c4f8a",
"#3b518b", "#39558b", "#37598c", "#365c8c", "#34608c", "#33638d",
"#31678d", "#2f6b8d", "#2d6e8e", "#2c718e", "#2b748e", "#29788e",
"#287c8e", "#277f8e", "#25848d", "#24878d", "#238b8d", "#218f8d",
"#21918d", "#22958b", "#23988a", "#239b89", "#249f87", "#25a186",
"#25a584", "#26a883", "#27ab82", "#29ae80", "#2eb17d", "#35b479",
"#3cb875", "#42bb72", "#49be6e", "#4ec16b", "#55c467", "#5cc863",
"#61c960", "#6bcc5a", "#72ce55", "#7cd04f", "#85d349", "#8dd544",
"#97d73e", "#9ed93a", "#a8db34", "#b0dd31", "#b8de30", "#c3df2e",
"#cbe02d", "#d6e22b", "#e1e329", "#eae428", "#f5e626", "#fde725"]



def display_eval_metrics(value):

    ### Comparison of Possible Models
    if value==choices[0]:
        compare_models=pd.read_csv('resources/compare_models_cust.csv', index_col=0)
        mydata1 = go.Bar(
            x=compare_models.loc['F1 score'].index,
            y=compare_models.loc['F1 score'],
            name=compare_models.index[0],
            marker=dict(color=Viridis[50])
        )
        mydata2 = go.Bar(
            x=compare_models.loc['Accuracy'].index,
            y=compare_models.loc['Accuracy'],
            name=compare_models.index[1],
            marker=dict(color=Viridis[30])
        )
        mydata3 = go.Bar(
            x=compare_models.loc['AUC score'].index,
            y=compare_models.loc['AUC score'],
            name=compare_models.index[2],
            marker=dict(color=Viridis[10])
        )
        mylayout = go.Layout(
            title='Logistic Regression has the highest accuracy and ROC-AUC score',
            xaxis = dict(title = 'Predictive models'), # x-axis label
            yaxis = dict(title = 'Score'), # y-axis label

        )
        fig = go.Figure(data=[mydata1, mydata2, mydata3], layout=mylayout)
        return fig

    ### Final Model Metrics
    elif value==choices[1]:
        file = open('resources/eval_scores_cust.pkl', 'rb')
        evals=pickle.load(file)
        file.close()
        mydata = [go.Bar(
            x=list(evals.keys()),
            y=list(evals.values()),
            marker=dict(color=Viridis[::12])
        )]

        mylayout = go.Layout(
            title='Evaluation Metrics for Logistic Regression Model (Testing Dataset = 443 rows)',
            xaxis = {'title': 'Metrics'},
            yaxis = {'title': 'Percent'},

        )
        fig = go.Figure(data=mydata, layout=mylayout)
        return fig

    # Receiver Operating Characteristic (ROC): Area Under Curve
    elif value==choices[2]:

        with open('resources/roc_dict_cust.json') as json_file:
            roc_dict = json.load(json_file)
        FPR=roc_dict['FPR']
        TPR=roc_dict['TPR']
        y_test=pd.Series(roc_dict['y_test'])
        predictions=roc_dict['predictions']

        roc_score=round(100*roc_auc_score(y_test, predictions),1)
        trace0=go.Scatter(
                x=FPR,
                y=TPR,
                mode='lines',
                name=f'AUC: {roc_score}',
                marker=dict(color=Viridis[10])
                )
        trace1=go.Scatter(
                x=[0,1],
                y=[0,1],
                mode='lines',
                name='Baseline Area: 50.0',
            marker=dict(color=Viridis[50])
                )
        layout=go.Layout(
            title='Receiver Operating Characteristic (ROC): Area Under Curve',
            xaxis={'title': 'False Positive Rate (100-Specificity)','scaleratio': 1,'scaleanchor': 'y'},
            yaxis={'title': 'True Positive Rate (Sensitivity)'}
            )
        data=[trace0, trace1]
        fig = dict(data=data, layout=layout)
        return fig

    # Confusion Matrix
    elif value==choices[3]:
        with open('resources/roc_dict_cust.json') as json_file:
            roc_dict = json.load(json_file)
        FPR=roc_dict['FPR']
        TPR=roc_dict['TPR']
        y_test=pd.Series(roc_dict['y_test'])
        
        cm=pd.read_csv('resources/confusion_matrix_cust.csv')
        trace = go.Table(
            header=dict(values=cm.columns,
                        line = dict(color='#7D7F80'),
                        fill = dict(color=Viridis[55]),
                        align = ['left'] * 5),
            cells=dict(values=[cm[f'n={len(y_test)}'], cm['pred: ignored'], cm['pred: responded']],
                       line = dict(color='#7D7F80'),
                       fill = dict(color='white'),
                       align = ['left'] * 5))

        layout = go.Layout(
            title = f'Logistic Regression Model (Testing Dataset)',
        )

        fig = dict(data=[trace], layout=layout)
        return fig

    # Odds of Survival (Coefficients)
    elif value==choices[4]:
        coeffs=pd.read_csv('resources/coefficients_cust.csv')
        mydata = [go.Bar(
            x=coeffs['feature'],
            y=coeffs['coefficient'],
            marker=dict(color=Viridis[::-6])
        )]
        mylayout = go.Layout(
            title='Customer Behavior prediction',
            xaxis = {'title': 'Customer Features'},
            yaxis = {'title': 'Odds of Response'},

        )
        fig = go.Figure(data=mydata, layout=mylayout)
        return fig
