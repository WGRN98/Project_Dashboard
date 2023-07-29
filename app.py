#Run data.py for new plots and values
#Every plot had to be remade with plotly, unsure why

#Could not figure out how to use multi dropdown or multi select menus

#Now dash and site work much faster than last time

import pandas as pd
import pickle as pkl
import matplotlib.pyplot as plt
import matplotlib.ticker
import plotly.express as px
import plotly.graph_objects as go


#Importing Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

#Tabs styles, will be using the same as last time
tabs_styles = {
    'height': '44px',
    'align-items': 'center'
}
tab_style = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
    'border-radius': '15px',
    'background-color': '#F2F2F2',
    'box-shadow': '4px 4px 4px 4px lightgrey',
}
tab_selected_style = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px',
    'border-radius': '15px',
}


#loading data
data=pd.read_csv('data_dash.csv')
#print(data.info())
data['time']=pd.to_datetime(data['time'])
data=data.set_index('time', drop=True)
#print(data.info())
cols=[14,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
data_gen=data.drop(data.columns[cols],axis=1)
cols=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31]
data_load=data.drop(data.columns[cols],axis=1)
weather=pd.read_csv('weather_final.csv')
weather['time']=pd.to_datetime(weather['time'])
weather=weather.set_index('time', drop=True)
#print(data.info())
#print(data_gen.info())
#print(data_load.info())
#print(weather.info())

#loading images
fig_d1 = pkl.load(open('plot_load','rb'))
fig_d2 = pkl.load(open('plot_generation','rb'))
fig_d3 = pkl.load(open('plot_price','rb'))
fig_e = pkl.load(open('elbow','rb'))
fig_c1 = pkl.load(open('cluster_1','rb'))
fig_c2 = pkl.load(open('cluster_2','rb'))
fig_c3 = pkl.load(open('cluster_3','rb'))
fig_c6 = pkl.load(open('cluster_6','rb'))
fig_c7 = pkl.load(open('cluster_7','rb'))
fig_c8 = pkl.load(open('cluster_8','rb'))
fig_c9 = pkl.load(open('cluster_9','rb'))
fig_pl1_1 = pkl.load(open('fore_pl1_1','rb'))
fig_pl1_2 = pkl.load(open('fore_pl1_2','rb'))
fig_pl2_1 = pkl.load(open('fore_pl2_1','rb'))
fig_pl2_2 = pkl.load(open('fore_pl2_2','rb'))
fig_pl3_1 = pkl.load(open('fore_pl3_1','rb'))
fig_pl3_2 = pkl.load(open('fore_pl3_2','rb'))
fig_pl4_1 = pkl.load(open('fore_pl4_1','rb'))
fig_pl4_2 = pkl.load(open('fore_pl4_2','rb'))
fig_pl5_1 = pkl.load(open('fore_pl5_1','rb'))
fig_pl5_2 = pkl.load(open('fore_pl5_2','rb'))
fig_pl6_1 = pkl.load(open('fore_pl6_1','rb'))
fig_pl6_2 = pkl.load(open('fore_pl6_2','rb'))
fig_pl7_1 = pkl.load(open('fore_pl7_1','rb'))
fig_pl7_2 = pkl.load(open('fore_pl7_2','rb'))
fig_pl8_1 = pkl.load(open('fore_pl8_1','rb'))
fig_pl8_2 = pkl.load(open('fore_pl8_2','rb'))
fig_pl9_1 = pkl.load(open('fore_pl9_1','rb'))
fig_pl9_2 = pkl.load(open('fore_pl9_2','rb'))
fig_pg1_1 = pkl.load(open('fore_pg1_1','rb'))
fig_pg1_2 = pkl.load(open('fore_pg1_2','rb'))
fig_pg2_1 = pkl.load(open('fore_pg2_1','rb'))
fig_pg2_2 = pkl.load(open('fore_pg2_2','rb'))
fig_pg3_1 = pkl.load(open('fore_pg3_1','rb'))
fig_pg3_2 = pkl.load(open('fore_pg3_2','rb'))
fig_pg4_1 = pkl.load(open('fore_pg4_1','rb'))
fig_pg4_2 = pkl.load(open('fore_pg4_2','rb'))
fig_pg5_1 = pkl.load(open('fore_pg5_1','rb'))
fig_pg5_2 = pkl.load(open('fore_pg5_2','rb'))
fig_pg6_1 = pkl.load(open('fore_pg6_1','rb'))
fig_pg6_2 = pkl.load(open('fore_pg6_2','rb'))
fig_pg7_1 = pkl.load(open('fore_pg7_1','rb'))
fig_pg7_2 = pkl.load(open('fore_pg7_2','rb'))
fig_pg8_1 = pkl.load(open('fore_pg8_1','rb'))
fig_pg8_2 = pkl.load(open('fore_pg8_2','rb'))
fig_pg9_1 = pkl.load(open('fore_pg9_1','rb'))
fig_pg9_2 = pkl.load(open('fore_pg9_2','rb'))
fig_pp1_1 = pkl.load(open('fore_pp1_1','rb'))
fig_pp1_2 = pkl.load(open('fore_pp1_2','rb'))
fig_pp2_1 = pkl.load(open('fore_pp2_1','rb'))
fig_pp2_2 = pkl.load(open('fore_pp2_2','rb'))
fig_pp3_1 = pkl.load(open('fore_pp3_1','rb'))
fig_pp3_2 = pkl.load(open('fore_pp3_2','rb'))
fig_pp4_1 = pkl.load(open('fore_pp4_1','rb'))
fig_pp4_2 = pkl.load(open('fore_pp4_2','rb'))
fig_pp5_1 = pkl.load(open('fore_pp5_1','rb'))
fig_pp5_2 = pkl.load(open('fore_pp5_2','rb'))
fig_pp6_1 = pkl.load(open('fore_pp6_1','rb'))
fig_pp6_2 = pkl.load(open('fore_pp6_2','rb'))
fig_pp7_1 = pkl.load(open('fore_pp7_1','rb'))
fig_pp7_2 = pkl.load(open('fore_pp7_2','rb'))
fig_pp8_1 = pkl.load(open('fore_pp8_1','rb'))
fig_pp8_2 = pkl.load(open('fore_pp8_2','rb'))
fig_pp9_1 = pkl.load(open('fore_pp9_1','rb'))
fig_pp9_2 = pkl.load(open('fore_pp9_2','rb'))



#Generate table function
def generate_table(dataframe, max_rows=100):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div([
    html.Img(src=app.get_asset_url('IST_Logo.png'), style={"display": "flex",
                               "justifyContent": "left", "height": "5%", "width": "5%"}),
    html.H1('Spanish Consumption, Generation and Energy Price Predictions', style={"display": "flex",
                               "justifyContent": "center"}),
    html.H5('by William Narciso - 84421'),

html.Div([
    html.Div([
        dcc.Tabs(id = "tabs-styled-with-inline", value = 'tab-1', children = [
            dcc.Tab(label = 'Initial Data', value = 'tab-1', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Clustering', value = 'tab-2', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Load Forecast', value = 'tab-3', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Generation Forecast', value = 'tab-4', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Price Prediction', value = 'tab-5', style = tab_style, selected_style = tab_selected_style),
        ], style = tabs_styles),
        html.Div(id = 'tabs-content-inline')
    ], className = "create_container3 eight columns", ),
    ], className = "row flex-display"),
])

@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))

def render_content(tab):
    if tab == 'tab-1':
         return html.Div([
                    html.Div([
                        html.P('Data on Spanish consumption, generation and energy price', className = 'fix_label', style = {'color': 'black'}),
                        dcc.RadioItems(id = 'radio_data',
                                   labelStyle = {"display": "inline-block"},
                                   options = [
                                       {'label': 'Consumption', 'value': 1},
                                       {'label': 'Generation', 'value': 2},
                                       {'label': 'Price', 'value': 3},
                                       {'label': 'Load Table', 'value': 4},
                                       {'label': 'Generation Table', 'value': 5},
                                       {'label': 'Weather Table', 'value': 6}],
                                   value = 1,
                                   style = {'text-align': 'left', 'color': 'black'}, className = 'dcc_compon'),
                                html.Div(id='data_info'),
                    ])
                ])
        
     
    elif tab == 'tab-2':
         return html.Div([
                    html.Div([
                        html.P('Clustering Data', className = 'fix_label', style = {'color': 'black'}),
                        dcc.RadioItems(id = 'cluster_select',
                                   labelStyle = {"display": "inline-block"},
                                   options = [
                                       {'label': 'Elbow Curve', 'value': 1},
                                       {'label': 'Load x Generation', 'value': 2},
                                       {'label': 'Price x Load', 'value': 3},
                                       {'label': 'Price x Generation', 'value': 4},
                                       {'label': 'Hour x Load', 'value': 5},
                                       {'label': 'Hour x Generation', 'value': 6},
                                       {'label': 'Hour x Solar Generation', 'value': 7},
                                       {'label': 'Wind Speed x Onshore Wind Generation', 'value': 8}],
                                   value = 1,
                                   style = {'text-align': 'left', 'color': 'black'}, className = 'dcc_compon'),
                                html.Div(id='clustering'),
                    ])
                ])
     
    
    elif tab == 'tab-3':
        return html.Div([
                         dcc.Dropdown(
                             id='method load',
                             options=[
                                 {'label': 'Linear Regression', 'value': 0},
                                 {'label': 'Support vector Regressor', 'value': 1},
                                 {'label': 'Decision Tree Regressor', 'value': 2},
                                 {'label': 'Random Forest', 'value': 3},
                                 {'label': 'Uniformised Data', 'value': 4},
                                 {'label': 'Gradient Boosting', 'value': 5},
                                 {'label': 'Extreme Gradient Boosting', 'value': 6},
                                 {'label': 'Bootstrapping', 'value': 7},
                                 {'label': 'Neural Networks', 'value': 8}
                                 ],
                             value=0,
                                 ),
                                html.Div(id='forecast load'),
                    ])
    
    elif tab == 'tab-4':
        return html.Div([
                         dcc.Dropdown(
                             id='method gen',
                             options=[
                                 {'label': 'Linear Regression', 'value': 0},
                                 {'label': 'Support vector Regressor', 'value': 1},
                                 {'label': 'Decision Tree Regressor', 'value': 2},
                                 {'label': 'Random Forest', 'value': 3},
                                 {'label': 'Uniformised Data', 'value': 4},
                                 {'label': 'Gradient Boosting', 'value': 5},
                                 {'label': 'Extreme Gradient Boosting', 'value': 6},
                                 {'label': 'Bootstrapping', 'value': 7},
                                 {'label': 'Neural Networks', 'value': 8}
                                 ],
                             value=0,
                                 ),
                                html.Div(id='forecast gen'),
                    ])
        
    elif tab == 'tab-5':
        return html.Div([
                         dcc.Dropdown(
                             id='method price',
                             options=[
                                 {'label': 'Linear Regression', 'value': 0},
                                 {'label': 'Support vector Regressor', 'value': 1},
                                 {'label': 'Decision Tree Regressor', 'value': 2},
                                 {'label': 'Random Forest', 'value': 3},
                                 {'label': 'Uniformised Data', 'value': 4},
                                 {'label': 'Gradient Boosting', 'value': 5},
                                 {'label': 'Extreme Gradient Boosting', 'value': 6},
                                 {'label': 'Bootstrapping', 'value': 7},
                                 {'label': 'Neural Networks', 'value': 8}
                                 ],
                             value=0,
                                 ),
                                html.Div(id='forecast price'),
                    ])
    
@app.callback(Output('data_info', 'children'), 
              Input('radio_data', 'value'))

def render_table(radio_value):   
    if radio_value == 1:
        return html.Div([dcc.Graph(figure=fig_d1)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 2:
        return html.Div([dcc.Graph(figure=fig_d2)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 3:
        return html.Div([dcc.Graph(figure=fig_d3)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 4:
        return generate_table(data_load)
    elif radio_value == 5:
        return generate_table(data_gen)
    elif radio_value == 6:
        return generate_table(weather)
    
@app.callback(Output('clustering', 'children'), 
              Input('cluster_select', 'value'))

def render_table(radio_value):   
    if radio_value == 1:
        return html.Div([dcc.Graph(figure=fig_e)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 2:
        return html.Div([dcc.Graph(figure=fig_c1)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 3:
        return html.Div([dcc.Graph(figure=fig_c2)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 4:
        return html.Div([dcc.Graph(figure=fig_c3)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 5:
        return html.Div([dcc.Graph(figure=fig_c6)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 6:
        return html.Div([dcc.Graph(figure=fig_c7)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 7:
        return html.Div([dcc.Graph(figure=fig_c8)],style={'display': 'inline-block', 'padding': '0 20'}),
    elif radio_value == 8:
        return html.Div([dcc.Graph(figure=fig_c9)],style={'display': 'inline-block', 'padding': '0 20'}),
    
@app.callback(Output('forecast load', 'children'), 
              Input('method load', 'value'))

def scatter_plot(method_value):
    if method_value == 0:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl1_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl1_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 1:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl2_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl2_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 2:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl3_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl3_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 3:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl4_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl4_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 4:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl5_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl5_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    if method_value == 5:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl6_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl6_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 6:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl7_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl7_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 7:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl8_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl8_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 8:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pl9_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pl9_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])



@app.callback(Output('forecast gen', 'children'), 
              Input('method gen', 'value'))

def scatter_plot(method_value):
    if method_value == 0:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg1_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg1_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 1:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg2_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg2_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 2:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg3_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg3_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 3:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg4_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg4_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 4:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg5_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg5_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    if method_value == 5:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg6_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg6_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 6:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg7_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg7_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 7:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg8_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg8_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 8:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pg9_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pg9_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])



@app.callback(Output('forecast price', 'children'), 
              Input('method price', 'value'))

def scatter_plot(method_value):
    if method_value == 0:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp1_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp1_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 1:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp2_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp2_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 2:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp3_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp3_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 3:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp4_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp4_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 4:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp5_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp5_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    if method_value == 5:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp6_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp6_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 6:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp7_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp7_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 7:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp8_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp8_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])
    
    if method_value == 8:
        return html.Div([
            html.Div([dcc.Graph(figure=fig_pp9_1)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Div([dcc.Graph(figure=fig_pp9_2)],style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            ])


if __name__ == '__main__':
    app.run_server(debug=False)
