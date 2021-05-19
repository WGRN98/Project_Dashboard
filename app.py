#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import plotly.express as px

#Importing csv data files
raw_consumption_2017=pd.read_csv('IST_South_Tower_2017_Ene_Cons.csv')
raw_consumption_2018=pd.read_csv('IST_South_Tower_2018_Ene_Cons.csv')
raw_holiday=pd.read_csv('holiday_17_18_19.csv')
raw_meteo=pd.read_csv('IST_meteo_data_2017_2018_2019.csv')

#Changing some column names for easier use
raw_consumption_2017=raw_consumption_2017.rename(columns={'Date_start' : 'Date'})
raw_consumption_2018=raw_consumption_2018.rename(columns={'Date_start' : 'Date'})
raw_holiday=raw_holiday.rename(columns={'Date_start' : 'Date'})
raw_meteo=raw_meteo.rename(columns={'yyyy-mm-dd hh:mm:ss' : 'Date'})

#Converting time objects to datetime format and setting datetime as index
raw_consumption_2017['Date']=pd.to_datetime(raw_consumption_2017['Date'])
raw_consumption_2018['Date']=pd.to_datetime(raw_consumption_2018['Date'])
raw_holiday['Date']=pd.to_datetime(raw_holiday['Date'])
raw_meteo['Date']=pd.to_datetime(raw_meteo['Date'])

#Setting datetime as index
df_consumption_2017=raw_consumption_2017.set_index('Date', drop=True)
df_consumption_2018=raw_consumption_2018.set_index('Date', drop=True)
df_holiday=raw_holiday.set_index('Date', drop=True)
df_meteo=raw_meteo.set_index('Date', drop=True)

#Resampling for merging of data files
#Downsampling meteo files to hour using mean and upsampling holiday file also to hour and filling non-holidays with 0
df_meteo=df_meteo.resample('H').mean()
df_holiday=df_holiday.resample('D').asfreq()
df_holiday['Holiday']=df_holiday['Holiday'].fillna(0)
df_holiday=df_holiday.resample('H').pad()

#File merging for easier use
df_data_2017=df_consumption_2017.merge(df_holiday, left_index=True, right_index=True, how='left')
df_data_2017=df_data_2017.merge(df_meteo, left_index=True, right_index=True, how='inner')
df_data_2018=df_consumption_2018.merge(df_holiday, left_index=True, right_index=True, how='left')
df_data_2018=df_data_2018.merge(df_meteo, left_index=True, right_index=True, how="inner")
df_data_2019=df_holiday.merge(df_meteo, left_index=True, right_index=True, how='left')
df_data_2019=df_data_2019.truncate(before=pd.Timestamp('2019-01-01'), after=None)

df_data_2017['Weekday']=df_data_2017.index.dayofweek
df_data_2017['Hour']=df_data_2017.index.hour
df_data_2017['Power-1']=df_data_2017['Power_kW'].shift(1)#Previous hour consumption
df_data_2018['Weekday']=df_data_2018.index.dayofweek
df_data_2018['Hour']=df_data_2018.index.hour
df_data_2018['Power-1']=df_data_2018['Power_kW'].shift(1)#Previous hour consumption
df_data_2019['Weekday']=df_data_2019.index.dayofweek
df_data_2019['Hour']=df_data_2019.index.hour

df_data=pd.concat([df_data_2017, df_data_2018])

#Deleting 0 Power values that are not possible
df_data=df_data[df_data.Power_kW != 0]
#Deleting single extremely high energy point that is clearly an outlier
df_data=df_data[df_data.Power_kW < 850]
#There is a gap in the meteo data in 2018, the remaining data is enough for prediction
df_data=df_data.dropna()
#Holiday/weekday, first day is Saturday, so saturdays and Holidays are a 0 and Sunday is a 1
df_data['Holtimesweek']=((df_data['Weekday']+2)%7)*np.abs(1-df_data['Holiday'])
#Heating degree.hour
df_data['HDH']=np.maximum(0,df_data['temp_C']-16)
#Temp * Solar_rad
df_data['temp_rad']=df_data['temp_C']*df_data['solarRad_W/m2']
#Weekday square
df_data['day2']=np.square(df_data['Weekday'])
#HDH/Rad
df_data['HDH_Rad']=df_data['HDH']*df_data['solarRad_W/m2']
#Month, forgot about it before, could be important
df_data['Month']=df_data.index.month

#print(df_data.info())

#Deleting 0 Power values that are not possible
df_data=df_data[df_data.Power_kW != 0]
#Deleting single extremely high energy point that is clearly an outlier
df_data=df_data[df_data.Power_kW < 850]
#There is a gap in the meteo data in 2018, the remaining data is enough for prediction
df_data=df_data.dropna()
df_data=df_data[df_data['Power_kW'] > df_data['Power_kW'].quantile(0.025)]

#Power(kW) plot again
fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
fig.set_size_inches(15,10) # define figure size
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
ax.xaxis.set_tick_params(which = 'major', pad = 4, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
plt.plot (df_data['Power_kW'], '-o', color = 'blue', # x axis laels; data; symbol type *try '-p'; line color;
         markersize = 10, linewidth = 0.4, # point size; line thickness;
         markerfacecolor = 'cyan', # color inside the point
         markeredgecolor = 'brown', # color of edge
         markeredgewidth = 3)
plt.savefig('assets/fig_plot.png')
plt.show()

print(df_data.info())

#Feature selection
from sklearn.feature_selection import SelectKBest #selection method
from sklearn.feature_selection import f_regression, mutual_info_regression #score metric
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

#Transforming dataframes into arrays and defining inputs and outputs
X=df_data.values
Y_f=X[:,0]
X_f=X[:,[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18]]

features_regression=SelectKBest(k=3, score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit_regression=features_regression.fit(X_f,Y_f) #Calculates the f_regression of the features
print(fit_regression.scores_)
features_regression_results=fit_regression.transform(X_f)
#print(features_regression_results)

features_mutual=SelectKBest(k=3, score_func=mutual_info_regression) #Test diferent k number of features
fit_mutual=features_mutual.fit(X_f,Y_f) #Calculates the f_regression of the features
print(fit_mutual.scores_)
features_mutual_results=fit_mutual.transform(X_f)
#print(features_mutual_results)

model_linear=LinearRegression() #LinearRegression Model as estimator
rfe=RFE(model_linear,n_features_to_select=1) #Using 1 features
fit_linear=rfe.fit(X_f,Y_f)
print("Feature Ranking (Linear Model, 1 features): %s" % (fit_linear.ranking_)) 

model_forest=RandomForestRegressor()
model_forest.fit(X_f,Y_f)
print(model_forest.feature_importances_)



#Sorry about this, only way I could make it work
selection = [['Holiday', 1.15918777e+02, 0.00756602, 5, 6.01242865e-04],
             ['Temperature', 2.11047004e+03, 0.1176159, 10, 9.87339803e-03],
             ['Relative Humidity', 1.77622412e+03, 0.08783419, 15, 8.88992887e-03],
             ['Windspeed', 3.89267511e+01, 0.02406593, 1, 3.76926640e-03],
             ['Wind Gust', 4.68727825e+01, 0.02798786, 2, 3.42906994e-03],
             ['Pressure', 4.20068583e+01, 0.05677975, 13, 7.98047764e-03],
             ['Solar Radiation', 7.95079760e+03, 0.29962079, 16, 1.21238503e-02],
             ['Rain', 2.37796316e-01, 0.0017438, 8, 1.05040071e-03],
             ['Rainy Day', 3.61116505e+00, 0.04411203, 12, 1.16252756e-03],
             ['Weekday', 4.21035776e+02, 0.04740485, 9, 2.43248527e-03],
             ['Hour', 3.86055405e+02, 0.54089701, 6, 7.73410453e-02],
             ['Power-1', 6.14580772e+04, 1.17254857, 7, 8.35926125e-01],
             ['Holiday x Weekday', 6.72930049e+02, 0.0510903, 4, 3.42323868e-03],
             ['Heating Degree Hour (HDH)', 2.20123809e+03, 0.10167673, 3, 4.51643036e-03],
             ['Temperature x Radiation', 7.49494561e+03, 0.31654585, 17, 8.22367903e-03],
             ['Day Squared', 5.87055279e+02, 0.04739732, 14, 2.36707425e-03],
             ['HDH x Radiation', 4.05358376e+03, 0.18715072, 18, 1.02861413e-02],
             ['Month', 5.95582291e+01, 0.10575823, 11, 6.60361814e-03],
             ]
df_selection=pd.DataFrame(selection, columns=['Feature', 'Univariate linear regression', 'Estimate mutual information', 'Linear regression', 'Random forest regression'])



#Just for site presentation
df_data_final=df_data.drop(columns=['HR', 'windSpeed_m/s', 'windGust_m/s', 'pres_mbar', 'rain_mm/h', 'rain_day', 'Holiday', 'Weekday', 'Month', 'day2', 'HDH', 'HDH_Rad', 'temp_C', 'solarRad_W/m2'])

df_data_2017_final=df_data_final[:'2017-12-31']
df_data_2017_final.reset_index(inplace=True)
df_data_2018_final=df_data_final['2018-01-01':]
df_data_2018_final.reset_index(inplace=True)



#Prediction methods
#Regression
from sklearn.model_selection import train_test_split
from sklearn import metrics

Y=X[:,0]
X=X[:,[11,12,13,15]] 
X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)

#Importing methods
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor
#from sklearn.ensemble import RandomForestRegressor #Repeated
#from sklearn.preprocessing import StandardScaler #Repeated

#Linear Regression
#Create linear regression object
regr = linear_model.LinearRegression()
#Train the model using the training sets
regr.fit(X_train,y_train)
#Make predictions using the testing set
y_pred_LR = regr.predict(X_test)
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)

#Support Vector Regressor
sc_X = StandardScaler()
sc_y = StandardScaler()
X_train_SVR = sc_X.fit_transform(X_train)
y_train_SVR = sc_y.fit_transform(y_train.reshape(-1,1))
regr = SVR(kernel='rbf')
regr.fit(X_train_SVR,y_train_SVR)
y_pred_SVR = regr.predict(sc_X.fit_transform(X_test))
y_test_SVR=sc_y.fit_transform(y_test.reshape(-1,1))
y_pred_SVR2=sc_y.inverse_transform(y_pred_SVR)
y_pred_SVR = sc_y.inverse_transform(regr.predict(sc_X.fit_transform(X_test)))
#Evaluate errors
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)

#Decision Tree Regressor
#Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()
#Train the model using the training sets
DT_regr_model.fit(X_train, y_train)
#Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)
#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)

#Random Forest
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 200, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 20,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
RF_model = RandomForestRegressor()
RF_model.fit(X_train, y_train)
y_pred_RF = RF_model.predict(X_test)
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)

#Uniformised Data
scaler = StandardScaler()
#Fit only to the training data
scaler.fit(X_train)
#Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
parameters_2 = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}
RF_model_2 = RandomForestRegressor(**parameters_2)
RF_model_2.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RF_2 = RF_model_2.predict(X_test_scaled)
#Evaluate errors
MAE_RF_2=metrics.mean_absolute_error(y_test,y_pred_RF_2) 
MSE_RF_2=metrics.mean_squared_error(y_test,y_pred_RF_2)  
RMSE_RF_2= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF_2=RMSE_RF_2/np.mean(y_test)

#Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)
GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)
#Evaluate errors
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)

#Extreme Gradient Boosting
params_2 = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params_2)
XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)
#Evaluate errors
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)

#Bootstrapping
BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)
#Evaluate errors
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)

#Neural Networks
NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)
#Evaluate errors
MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)


plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.savefig('assets/fig0s.png')
plt.show()
plt.scatter(y_test,y_pred_LR)
plt.savefig('assets/fig0p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_SVR2[1:200])
plt.savefig('assets/fig1s.png')
plt.show()
plt.scatter(y_test,y_pred_SVR2)
plt.savefig('assets/fig1p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_DT[1:200])
plt.savefig('assets/fig2s.png')
plt.show()
plt.scatter(y_test,y_pred_DT)
plt.savefig('assets/fig2p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.savefig('assets/fig3s.png')
plt.show()
plt.scatter(y_test,y_pred_RF)
plt.savefig('assets/fig3p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.savefig('assets/fig4s.png')
plt.show()
plt.scatter(y_test,y_pred_RF)
plt.savefig('assets/fig4p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_GB[1:200])
plt.savefig('assets/fig5s.png')
plt.show()
plt.scatter(y_test,y_pred_GB)
plt.savefig('assets/fig5p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_XGB[1:200])
plt.savefig('assets/fig6s.png')
plt.show()
plt.scatter(y_test,y_pred_XGB)
plt.savefig('assets/fig6p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_BT[1:200])
plt.savefig('assets/fig7s.png')
plt.show()
plt.scatter(y_test,y_pred_BT)
plt.savefig('assets/fig7p.png')
plt.show()

plt.plot(y_test[1:200])
plt.plot(y_pred_NN[1:200])
plt.savefig('assets/fig8s.png')
plt.show()
plt.scatter(y_test,y_pred_NN)
plt.savefig('assets/fig8p.png')
plt.show()

#Importing Dash
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

#Tabs styles, did not find in css file, adding manually
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


#Generate rable function
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


#Generate scatter plot

#Generate prediction plot

#Generate line plot

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


app.layout = html.Div((

    html.Div([
        html.Div([
            html.Div([
                html.H3('South Tower Power Consumption Dashboard', style = {'margin-bottom': '0px', 'color': 'black'}),
            ])
        ], className = "create_container1 four columns", id = "title"),

    ], id = "header", className = "row flex-display", style = {"margin-bottom": "25px"}),

html.Div([
    html.Div([
        dcc.Tabs(id = "tabs-styled-with-inline", value = 'tab-1', children = [
            dcc.Tab(label = 'Raw Data', value = 'tab-1', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Processed Data', value = 'tab-2', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Feature Selection', value = 'tab-3', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Final Data', value = 'tab-4', style = tab_style, selected_style = tab_selected_style),
            dcc.Tab(label = 'Forecasts', value = 'tab-5', style = tab_style, selected_style = tab_selected_style),
        ], style = tabs_styles),
        html.Div(id = 'tabs-content-inline')
    ], className = "create_container3 eight columns", ),
    ], className = "row flex-display"),
))


@app.callback(Output('tabs-content-inline', 'children'),
              Input('tabs-styled-with-inline', 'value'))

def render_content(tab):
    if tab == 'tab-1':
         return html.Div([
                    html.Div([
                        html.P('This data is just the raw data, not yet clean and incompatible with each other due to different time frames', className = 'fix_label', style = {'color': 'black'}),
                        dcc.RadioItems(id = 'radio_raw',
                                   labelStyle = {"display": "inline-block"},
                                   options = [
                                       {'label': '2017 Consumption', 'value': 2017},
                                       {'label': '2018 Consumption', 'value': 2018},
                                       {'label': 'Meteorological Data', 'value': 2019},#should be a string, but I'm not sure how to use them
                                       {'label': 'Holiday Data', 'value': 2020}],#should be a string, but I'm not sure how to use them
                                   value = 2017,
                                   style = {'text-align': 'left', 'color': 'black'}, className = 'dcc_compon'),
                                html.Div(id='raw_table'),
                    ])
                ])
        
    elif tab == 'tab-2':
        return html.Div([
                    html.Div([
                        html.P('Data after cleaning and joining and slight processing', className = 'fix_label', style = {'color': 'black'}),
                                html.Div(generate_table(df_data)),
                                
                    ])
                ])
    
    elif tab == 'tab-3':
        return html.Div([
                    html.Div([
                        html.P('Feature selection process', className = 'fix_label', style = {'color': 'black'}),
                                html.Div(generate_table(df_selection)),    
                    ])
                ])
    
    elif tab == 'tab-4':
         return html.Div([
                    html.Div([
                        html.P('Final data to be used for forecasts', className = 'fix_label', style = {'color': 'black'}),
                        dcc.RadioItems(id = 'radio_final',
                                   labelStyle = {"display": "inline-block"},
                                   options = [
                                       {'label': '2017 Data', 'value': 2017},
                                       {'label': '2018 Data', 'value': 2018}],
                                   value = 2017,
                                   style = {'text-align': 'left', 'color': 'black'}, className = 'dcc_compon'),
                                html.Div(id='final_table',style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),     
                                html.Img(src=app.get_asset_url('fig_plot.png'),style={'width': '50%', 'display': 'inline-block', 'vertical-align': 'top'}),
                    ])
                ])
        
    elif tab == 'tab-5':
        return html.Div([
                         dcc.Dropdown(
                             id='method',
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
                                html.Div(id='forecast'),
                    ])
                
             
@app.callback(Output('raw_table', 'children'), 
              Input('radio_raw', 'value'))

def render_table(radio_value):   
    if radio_value == 2017:
        return generate_table(raw_consumption_2017)
    elif radio_value == 2018:
        return generate_table(raw_consumption_2018)
    elif radio_value == 2019:
        return generate_table(raw_meteo)
    elif radio_value == 2020:
        return generate_table(raw_holiday)
    
    
@app.callback(Output('final_table', 'children'), 
              Input('radio_final', 'value'))

def render_table_final(radio_value):   
    if radio_value == 2017:
        return generate_table(df_data_2017_final)
    elif radio_value == 2018:
        return generate_table(df_data_2018_final)
    

@app.callback(Output('forecast', 'children'), 
              Input('method', 'value'))

def scatter_plot(method_value):
    if method_value == 0:
        return html.Div([
            html.Img(src=app.get_asset_url('fig0s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig0p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_LR]),
            html.P(children=["MSE = ", MSE_LR]),
            html.P(children=["RMSE = ", RMSE_LR]),
            html.P(children=["cvRMSE = ", cvRMSE_LR]),
            ])
    
    if method_value == 1:
        return html.Div([
            html.Img(src=app.get_asset_url('fig1s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig1p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_SVR]),
            html.P(children=["MSE = ", MSE_SVR]),
            html.P(children=["RMSE = ", RMSE_SVR]),
            html.P(children=["cvRMSE = ", cvRMSE_SVR]),
            ])
    
    if method_value == 2:
        return html.Div([
            html.Img(src=app.get_asset_url('fig2s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig2p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_DT]),
            html.P(children=["MSE = ", MSE_DT]),
            html.P(children=["RMSE = ", RMSE_DT]),
            html.P(children=["cvRMSE = ", cvRMSE_DT]),
            ])
    
    if method_value == 3:
        return html.Div([
            html.Img(src=app.get_asset_url('fig3s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig3p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_RF]),
            html.P(children=["MSE = ", MSE_RF]),
            html.P(children=["RMSE = ", RMSE_RF]),
            html.P(children=["cvRMSE = ", cvRMSE_RF]),
            ])
    
    if method_value == 4:
        return html.Div([
            html.Img(src=app.get_asset_url('fig4s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig4p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_RF_2]),
            html.P(children=["MSE = ", MSE_RF_2]),
            html.P(children=["RMSE = ", RMSE_RF_2]),
            html.P(children=["cvRMSE = ", cvRMSE_RF_2]),
            ])
    
    if method_value == 5:
        return html.Div([
            html.Img(src=app.get_asset_url('fig5s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig5p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_GB]),
            html.P(children=["MSE = ", MSE_GB]),
            html.P(children=["RMSE = ", RMSE_GB]),
            html.P(children=["cvRMSE = ", cvRMSE_GB]),
            ])
    
    if method_value == 6:
        return html.Div([
            html.Img(src=app.get_asset_url('fig6s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig6p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_XGB]),
            html.P(children=["MSE = ", MSE_XGB]),
            html.P(children=["RMSE = ", RMSE_XGB]),
            html.P(children=["cvRMSE = ", cvRMSE_XGB]),
            ])
    
    if method_value == 7:
        return html.Div([
            html.Img(src=app.get_asset_url('fig7s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig7p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_BT]),
            html.P(children=["MSE = ", MSE_BT]),
            html.P(children=["RMSE = ", RMSE_BT]),
            html.P(children=["cvRMSE = ", cvRMSE_BT]),
            ])
    
    if method_value == 8:
        return html.Div([
            html.Img(src=app.get_asset_url('fig8s.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.Img(src=app.get_asset_url('fig8p.png'),style={'width': '50%', 'display': 'inline-block', 'padding': '0 20'}),
            html.P(children=["MAE = ", MAE_NN]),
            html.P(children=["MSE = ", MSE_NN]),
            html.P(children=["RMSE = ", RMSE_NN]),
            html.P(children=["cvRMSE = ", cvRMSE_NN]),
            ])

if __name__ == '__main__':
    app.run_server(debug=True)
