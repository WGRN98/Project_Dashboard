#CAUTION, program takes a very long time to run, it is recommended that it be run in parts
#Skipping feature selection is recommended for faster run times

#Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import pickle as pkl #Needed to save images as python object to use in seperate dash file
import plotly.graph_objects as go
import plotly.express as px
#Graphs had to be remake in plotly to work with dash dcc.Graph

#import plotly.tools as tls
#Ployly tools no longer supports converting from matplotlib to plotly formats
#I didn't loose my mind at all with this

#Feature selection and prediction methods
from sklearn.feature_selection import SelectKBest #selection method
from sklearn.feature_selection import f_regression, mutual_info_regression #score metric
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import linear_model
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.neural_network import MLPRegressor

#Prediction methods
from sklearn.model_selection import train_test_split
from sklearn import metrics


#Importing and analysing data
energy=pd.read_csv('energy.csv')
#Weather was big, and using to much time so it has been completed and averaged beforehand, check weather fixing folder
weather=pd.read_csv('weather_final.csv')

print(energy.info())
print(weather.info())
#print(energy)
#print(weather)


#There is no fossil coal-derived gas, fossil oil shale, fossil peat, geothermal, marine or offshore wind generation
#Hydro pumped storage aggregated column is empty in every row
energy=energy.drop(columns=['generation fossil coal-derived gas','generation fossil oil shale','generation fossil peat','generation geothermal','generation marine','generation wind offshore','generation hydro pumped storage aggregated'])


#Converting time objects to datetime format and setting datetime as index
#All data is hourly
energy['time']=pd.to_datetime(energy['time'])
energy=energy.set_index('time', drop=True)
weather['time']=pd.to_datetime(weather['time'])
weather=weather.set_index('time', drop=True)

#Total generation
energy['total generation']=energy['generation biomass']+energy['generation fossil brown coal/lignite']+energy['generation fossil gas']+energy['generation fossil hard coal']+energy['generation fossil oil']+energy['generation hydro pumped storage consumption']+energy['generation hydro run-of-river and poundage']+energy['generation hydro water reservoir']+energy['generation nuclear']+energy['generation other']+energy['generation other renewable']+energy['generation solar']+energy['generation waste']+energy['generation wind onshore']

print(energy.info())
print(weather.info())
#print(energy)
#print(weather)


#Total load graph
fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
fig.set_size_inches(15,10) # define figure size
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
ax.xaxis.set_tick_params(which = 'major', pad = 4, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
plt.plot (energy['total load'], '-o', color = 'blue', # x axis laels; data; symbol type *try '-p'; line color;
         markersize = 10, linewidth = 0.4, # point size; line thickness;
         markerfacecolor = 'cyan', # color inside the point
         markeredgecolor = 'brown', # color of edge
         markeredgewidth = 3)
plt.show()
#plotly_fig=


#Total generation graph
fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
fig.set_size_inches(15,10) # define figure size
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
ax.xaxis.set_tick_params(which = 'major', pad = 4, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
plt.plot (energy['total generation'], '-o', color = 'blue', # x axis laels; data; symbol type *try '-p'; line color;
         markersize = 10, linewidth = 0.4, # point size; line thickness;
         markerfacecolor = 'cyan', # color inside the point
         markeredgecolor = 'brown', # color of edge
         markeredgewidth = 3)
plt.show()


#There is a variation of around 30 non-null rows in the energy columns, not a significant number compared to 35000 entries
energy=energy.dropna()
weather=weather.dropna()

#Merging the data
data=energy.merge(weather, left_index=True, right_index=True, how='inner')

print(data.info())

#Removing outliers with zscore method, using total generation for calculations, as it has more variations than total loads
from scipy import stats
z=np.abs(stats.zscore(data['total generation']))
threshold=3
data=data[(z<3)]


#Total load graph
fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
fig.set_size_inches(15,10) # define figure size
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
ax.xaxis.set_tick_params(which = 'major', pad = 4, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
plt.plot (data['total load'], '-o', color = 'blue', # x axis laels; data; symbol type *try '-p'; line color;
         markersize = 10, linewidth = 0.4, # point size; line thickness;
         markerfacecolor = 'cyan', # color inside the point
         markeredgecolor = 'brown', # color of edge
         markeredgewidth = 3)
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=data.index, y=data['total load'],
                    mode='lines+markers'))
pkl.dump(plotly_fig,open('assets/plot_load','wb'))

#Total generation graph
fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
fig.set_size_inches(15,10) # define figure size
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
ax.xaxis.set_tick_params(which = 'major', pad = 4, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
plt.plot (data['total generation'], '-o', color = 'blue', # x axis laels; data; symbol type *try '-p'; line color;
         markersize = 10, linewidth = 0.4, # point size; line thickness;
         markerfacecolor = 'cyan', # color inside the point
         markeredgecolor = 'brown', # color of edge
         markeredgewidth = 3)
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=data.index, y=data['total generation'],
                    mode='lines+markers'))
pkl.dump(plotly_fig,open('assets/plot_generation','wb'))

#Total generation graph
fig, ax = plt.subplots() # create objects of the plot (figure and plot inside)
fig.set_size_inches(15,10) # define figure size
ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(60)) # define the interval between ticks on x axis 
    # Try changing (the number) to see what it does
ax.xaxis.set_tick_params(which = 'major', pad = 4, labelrotation = 50)
    # parameters of major labels of x axis: pad = distance to the axis;
plt.plot (data['price'], '-o', color = 'blue', # x axis laels; data; symbol type *try '-p'; line color;
         markersize = 10, linewidth = 0.4, # point size; line thickness;
         markerfacecolor = 'cyan', # color inside the point
         markeredgecolor = 'brown', # color of edge
         markeredgewidth = 3)
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=data.index, y=data['price'],
                    mode='lines+markers'))
pkl.dump(plotly_fig,open('assets/plot_price','wb'))


#Elbow curve for number of clusters
#This just takes a lot of time
from sklearn.cluster import KMeans

model = KMeans(n_clusters=2).fit(data)
pred = model.labels_
#print(pred)

Nc=range(1, 10)
kmeans = [KMeans(n_clusters=i) for i in Nc]
#print(kmeans)
score = [kmeans[i].fit(data).score(data) for i in range(len(kmeans))]
#print(score)

fig=plt.plot(Nc,score)
plt.xlabel('Number of Clusters')
plt.ylabel('Score')
plt.title('Elbow Curve')
plt.show()
plotly_fig = px.line(Nc, score, title='Elbow Curve')
pkl.dump(plotly_fig,open('assets/elbow','wb'))
#I will be using just 2 clusters


#Some new features
data['load-1']=data['total load'].shift(1)
data['generation-1']=data['total generation'].shift(1)
data['hour']=data.index.hour
data['weekday']=data.index.dayofweek


#Clustering
data['cluster']=pred
#Still coundn't figure out how to use other colours
fig=data.plot.scatter(x='total load', y='total generation', color=data['cluster'])
plt.show()
plotly_fig=px.scatter(data, x="total load", y="total generation", color=data['cluster'])
pkl.dump(plotly_fig,open('assets/cluster_1','wb'))

fig=data.plot.scatter(x='price', y='total load', color=data['cluster'])
plt.show()
plotly_fig=px.scatter(data, x="price", y="total load", color=data['cluster'])
pkl.dump(plotly_fig,open('assets/cluster_2','wb'))

fig=data.plot.scatter(x='price', y='total generation', color=data['cluster'])
plt.show()
plotly_fig=px.scatter(data, x="price", y="total generation", color=data['cluster'])
pkl.dump(plotly_fig,open('assets/cluster_3','wb'))

fig=data.plot.scatter(x='hour', y='total load', color=data['cluster'])
plt.show()
plotly_fig=px.scatter(data, x="hour", y="total load", color=data['cluster'])
pkl.dump(plotly_fig,open('assets/cluster_6','wb'))

fig=data.plot.scatter(x='hour', y='total generation', color=data['cluster'])
plt.show()
plotly_fig=px.scatter(data, x="hour", y="total generation", color=data['cluster'])
pkl.dump(plotly_fig,open('assets/cluster_7','wb'))


#Some things that can be interesting
fig=data.plot.scatter(x='hour', y='generation solar', color=data['cluster'])
plt.show()
plotly_fig=px.scatter(data, x="hour", y="generation solar", color=data['cluster'])
pkl.dump(plotly_fig,open('assets/cluster_8','wb'))

fig=data.plot.scatter(x='wind_speed', y='generation wind onshore', color=data['cluster'])
plt.show()
plotly_fig=px.scatter(data, x="wind_speed", y="generation wind onshore", color=data['cluster'])
pkl.dump(plotly_fig,open('assets/cluster_9','wb'))


data=data.dropna()
print(data.info())



#Feature selection for load
#Not using individual generation data, only total
X=data.values
Y=X[:,14]
X=X[:,[15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
#print(Y)
#print(X)

#Filter methods
features=SelectKBest(k=3, score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #Calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)

features=SelectKBest(k=3, score_func=mutual_info_regression) #Test diferent k number of features
fit=features.fit(X,Y) #Calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)

#Wrapper methods
model=LinearRegression() #LinearRegression Model as estimator
rfe=RFE(model,n_features_to_select=1) #Using 1 features
rfe2=RFE(model,n_features_to_select=2) #Using 2 features

fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)
print("Feature Ranking (Linear Model, 1 features): %s" % (fit.ranking_)) 
print("Feature Ranking (Linear Model, 2 features): %s" % (fit2.ranking_))

#Emsemble methods
model=RandomForestRegressor()
model.fit(X,Y)
print(model.feature_importances_)

#Load was not very dependant on weather features
#It was the most dependant on load-1 as expected
#It was also dependant on hour
#Since we are using an entire country load during the day will be higher than at night
#Load was also dependant on generation and generation-1, as they are intertwined
#When load goes up so must generation to meet de energy demand
#Load was somewhat related to price, as energy can be more expensive at night and cheaper in the mornings




#Feature selection for generation
X=data.values
Y=X[:,16]
X=X[:,[14,15,17,18,19,20,21,22,23,24,25,26,27,28,29,30]]
#print(Y)
#print(X)

#Filter methods
features=SelectKBest(k=3, score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #Calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)

features=SelectKBest(k=3, score_func=mutual_info_regression) #Test diferent k number of features
fit=features.fit(X,Y) #Calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)

#Wrapper methods
model=LinearRegression() #LinearRegression Model as estimator
rfe=RFE(model,n_features_to_select=1) #Using 1 features
rfe2=RFE(model,n_features_to_select=2) #Using 2 features

fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)

print("Feature Ranking (Linear Model, 1 features): %s" % (fit.ranking_)) 
print("Feature Ranking (Linear Model, 2 features): %s" % (fit2.ranking_))

#Emsemble methods
model=RandomForestRegressor()
model.fit(X,Y)
print(model.feature_importances_)

#Generation also has the most correlation with generation-1
#Generation was more dependant upon weather features than load
#Spain has wind and solar energy generation that will depend on the weather
#Generation was a bit less dependant on price than load
#Hour is similarly correlated to generation and load




#Feature selection for price
X=data.values
Y=X[:,15]
X=X[:,[14,16,27,28,29,30]]
#print(Y)
#print(X)

#Filter methods
features=SelectKBest(k=3, score_func=f_regression) # Test different k number of features, uses f-test ANOVA
fit=features.fit(X,Y) #Calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)

features=SelectKBest(k=3, score_func=mutual_info_regression) #Test diferent k number of features
fit=features.fit(X,Y) #Calculates the f_regression of the features
print(fit.scores_)
features_results=fit.transform(X)
print(features_results)

#Wrapper methods
model=LinearRegression() #LinearRegression Model as estimator
rfe=RFE(model,n_features_to_select=1) #Using 1 features
rfe2=RFE(model,n_features_to_select=2) #Using 2 features

fit=rfe.fit(X,Y)
fit2=rfe2.fit(X,Y)

print("Feature Ranking (Linear Model, 1 features): %s" % (fit.ranking_)) 
print("Feature Ranking (Linear Model, 2 features): %s" % (fit2.ranking_))

#Emsemble methods
model=RandomForestRegressor()
model.fit(X,Y)
print(model.feature_importances_)

#Price was most dependant on load and load-1 instead of generation
#generation-1 practically doesn't affect price
#The day of the week also does not affect price much
#Price is somewhat dependant on hour, as load is also related to hour





#Predictions for load
#Splitting training data and test data
X=data.values
Y=X[:,0]
X=X[:,[15,28,27,29]] 
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
#Choosing 80/20 split as a starting point based on Pareto Principle
print(X_train)
print(y_train)


x_plot = list(range(0, 200))

#Linear Regression
#Create linear regression object
regr = linear_model.LinearRegression()
#Train the model using the training sets
regr.fit(X_train,y_train)
#Make predictions using the testing set
y_pred_LR = regr.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_LR[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl1_1','wb'))
fig=plt.scatter(y_test,y_pred_LR)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_LR)
pkl.dump(plotly_fig,open('assets/fore_pl1_2','wb'))
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)



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
#plt.plot(y_test_SVR[1:200])
#plt.plot(y_pred_SVR[1:200])
#plt.show()
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_SVR2[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_SVR2[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl2_1','wb'))
fig=plt.scatter(y_test,y_pred_SVR2)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_SVR2)
pkl.dump(plotly_fig,open('assets/fore_pl2_2','wb'))
#Evaluate errors
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
print(MAE_SVR, MSE_SVR, RMSE_SVR,cvRMSE_SVR)


#Decision Tree Regressor
#Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()
#Train the model using the training sets
DT_regr_model.fit(X_train, y_train)
#Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_DT[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_DT[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl3_1','wb'))
fig=plt.scatter(y_test,y_pred_DT)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_DT)
pkl.dump(plotly_fig,open('assets/fore_pl3_2','wb'))
#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
print(MAE_DT, MSE_DT, RMSE_DT,cvRMSE_DT)



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
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_RF[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl4_1','wb'))
fig=plt.scatter(y_test,y_pred_RF)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_RF)
pkl.dump(plotly_fig,open('assets/fore_pl4_2','wb'))
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)



#Uniformised Data
scaler = StandardScaler()
#Fit only to the training data
scaler.fit(X_train)
#Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RF = RF_model.predict(X_test_scaled)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_RF[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl5_1','wb'))
fig=plt.scatter(y_test,y_pred_RF)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_RF)
pkl.dump(plotly_fig,open('assets/fore_pl5_2','wb'))
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)



#Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)
GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_GB[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_GB[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl6_1','wb'))
fig=plt.scatter(y_test,y_pred_GB)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_GB)
pkl.dump(plotly_fig,open('assets/fore_pl6_2','wb'))
#Evaluate errors
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
print(MAE_GB,MSE_GB,RMSE_GB,cvRMSE_GB)



#Extreme Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)
XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_XGB[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_XGB[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl7_1','wb'))
fig=plt.scatter(y_test,y_pred_XGB)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_XGB)
pkl.dump(plotly_fig,open('assets/fore_pl7_2','wb'))
#Evaluate errors
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB)



#Bootstrapping
BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_BT[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_BT[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl8_1','wb'))
fig=plt.scatter(y_test,y_pred_BT)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_BT)
pkl.dump(plotly_fig,open('assets/fore_pl8_2','wb'))
#Evaluate errors
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
print(MAE_BT,MSE_BT,RMSE_BT,cvRMSE_BT)


#Neural Networks
NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_NN[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_NN[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pl9_1','wb'))
fig=plt.scatter(y_test,y_pred_NN)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_NN)
pkl.dump(plotly_fig,open('assets/fore_pl9_2','wb'))
#Evaluate errors
MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
print(MAE_NN,MSE_NN,RMSE_NN,cvRMSE_NN)








#Predictions for generation
#Splitting training data and test data
X=data.values
Y=X[:,0]
X=X[:,[28,14,15,29]] 
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
#Choosing 80/20 split as a starting point based on Pareto Principle
print(X_train)
print(y_train)

#Linear Regression
#Create linear regression object
regr = linear_model.LinearRegression()
#Train the model using the training sets
regr.fit(X_train,y_train)
#Make predictions using the testing set
y_pred_LR = regr.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_LR[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg1_1','wb'))
fig=plt.scatter(y_test,y_pred_LR)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_LR)
pkl.dump(plotly_fig,open('assets/fore_pg1_2','wb'))
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)



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
#plt.plot(y_test_SVR[1:200])
#plt.plot(y_pred_SVR[1:200])
#plt.show()
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_SVR2[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_SVR2[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg2_1','wb'))
fig=plt.scatter(y_test,y_pred_SVR2)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_SVR2)
pkl.dump(plotly_fig,open('assets/fore_pg2_2','wb'))
#Evaluate errors
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
print(MAE_SVR, MSE_SVR, RMSE_SVR,cvRMSE_SVR)


#Decision Tree Regressor
#Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()
#Train the model using the training sets
DT_regr_model.fit(X_train, y_train)
#Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_DT[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_DT[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg3_1','wb'))
fig=plt.scatter(y_test,y_pred_DT)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_DT)
pkl.dump(plotly_fig,open('assets/fore_pg3_2','wb'))
#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
print(MAE_DT, MSE_DT, RMSE_DT,cvRMSE_DT)



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
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_RF[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg4_1','wb'))
fig=plt.scatter(y_test,y_pred_RF)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_RF)
pkl.dump(plotly_fig,open('assets/fore_pg4_2','wb'))
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)



#Uniformised Data
scaler = StandardScaler()
#Fit only to the training data
scaler.fit(X_train)
#Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RF = RF_model.predict(X_test_scaled)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_RF[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg5_1','wb'))
fig=plt.scatter(y_test,y_pred_RF)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_RF)
pkl.dump(plotly_fig,open('assets/fore_pg5_2','wb'))
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)



#Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)
GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_GB[1:200])
plt.show()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_GB[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg6_1','wb'))
fig=plt.scatter(y_test,y_pred_GB)
plt.show()
plotly_fig=go.Figure()
plotly_fig=px.scatter(x=y_test, y=y_pred_GB)
pkl.dump(plotly_fig,open('assets/fore_pg6_2','wb'))
#Evaluate errors
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
print(MAE_GB,MSE_GB,RMSE_GB,cvRMSE_GB)



#Extreme Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)
XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_XGB[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_XGB[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg7_1','wb'))
fig=plt.scatter(y_test,y_pred_XGB)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_XGB)
pkl.dump(plotly_fig,open('assets/fore_pg7_2','wb'))
#Evaluate errors
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB)



#Bootstrapping
BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_BT[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_BT[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg8_1','wb'))
fig=plt.scatter(y_test,y_pred_BT)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_BT)
pkl.dump(plotly_fig,open('assets/fore_pg8_2','wb'))
#Evaluate errors
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
print(MAE_BT,MSE_BT,RMSE_BT,cvRMSE_BT)


#Neural Networks
NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_NN[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_NN[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pg9_1','wb'))
fig=plt.scatter(y_test,y_pred_NN)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_NN)
pkl.dump(plotly_fig,open('assets/fore_pg9_2','wb'))
#Evaluate errors
MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
print(MAE_NN,MSE_NN,RMSE_NN,cvRMSE_NN)






#Predictions for price
#Splitting training data and test data
X=data.values
Y=X[:,0]
X=X[:,[14,16,29]] 
#print(X)

X_train, X_test, y_train, y_test = train_test_split(X, Y, train_size=0.8, test_size=0.2)
#Choosing 80/20 split as a starting point based on Pareto Principle
print(X_train)
print(y_train)

#Linear Regression
#Create linear regression object
regr = linear_model.LinearRegression()
#Train the model using the training sets
regr.fit(X_train,y_train)
#Make predictions using the testing set
y_pred_LR = regr.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_LR[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_LR[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp1_1','wb'))
fig=plt.scatter(y_test,y_pred_LR)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_LR)
pkl.dump(plotly_fig,open('assets/fore_pp1_2','wb'))
#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y_test,y_pred_LR) 
MSE_LR=metrics.mean_squared_error(y_test,y_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y_test)
print(MAE_LR, MSE_LR, RMSE_LR,cvRMSE_LR)



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
#plt.plot(y_test_SVR[1:200])
#plt.plot(y_pred_SVR[1:200])
#plt.show()
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_SVR2[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_SVR2[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp2_1','wb'))
fig=plt.scatter(y_test,y_pred_SVR2)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_SVR2)
pkl.dump(plotly_fig,open('assets/fore_pp2_2','wb'))
#Evaluate errors
MAE_SVR=metrics.mean_absolute_error(y_test,y_pred_SVR2) 
MSE_SVR=metrics.mean_squared_error(y_test,y_pred_SVR2)  
RMSE_SVR= np.sqrt(metrics.mean_squared_error(y_test,y_pred_SVR2))
cvRMSE_SVR=RMSE_SVR/np.mean(y_test)
print(MAE_SVR, MSE_SVR, RMSE_SVR,cvRMSE_SVR)


#Decision Tree Regressor
#Create Regression Decision Tree object
DT_regr_model = DecisionTreeRegressor()
#Train the model using the training sets
DT_regr_model.fit(X_train, y_train)
#Make predictions using the testing set
y_pred_DT = DT_regr_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_DT[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_DT[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp3_1','wb'))
fig=plt.scatter(y_test,y_pred_DT)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_DT)
pkl.dump(plotly_fig,open('assets/fore_pp3_2','wb'))
#Evaluate errors
MAE_DT=metrics.mean_absolute_error(y_test,y_pred_DT) 
MSE_DT=metrics.mean_squared_error(y_test,y_pred_DT)  
RMSE_DT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_DT))
cvRMSE_DT=RMSE_DT/np.mean(y_test)
print(MAE_DT, MSE_DT, RMSE_DT,cvRMSE_DT)



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
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_RF[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp4_1','wb'))
fig=plt.scatter(y_test,y_pred_RF)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_RF)
pkl.dump(plotly_fig,open('assets/fore_pp4_2','wb'))
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)



#Uniformised Data
scaler = StandardScaler()
#Fit only to the training data
scaler.fit(X_train)
#Now apply the transformations to the data:
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
parameters = {'bootstrap': True,
              'min_samples_leaf': 3,
              'n_estimators': 100, 
              'min_samples_split': 15,
              'max_features': 'sqrt',
              'max_depth': 10,
              'max_leaf_nodes': None}
RF_model = RandomForestRegressor(**parameters)
RF_model.fit(X_train_scaled, y_train.reshape(-1,1))
y_pred_RF = RF_model.predict(X_test_scaled)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_RF[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_RF[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp5_1','wb'))
fig=plt.scatter(y_test,y_pred_RF)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_RF)
pkl.dump(plotly_fig,open('assets/fore_pp5_2','wb'))
#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y_test,y_pred_RF) 
MSE_RF=metrics.mean_squared_error(y_test,y_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y_test,y_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y_test)
print(MAE_RF,MSE_RF,RMSE_RF,cvRMSE_RF)



#Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)
GB_model = GradientBoostingRegressor()
GB_model.fit(X_train, y_train)
y_pred_GB =GB_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_GB[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_GB[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp6_1','wb'))
fig=plt.scatter(y_test,y_pred_GB)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_GB)
pkl.dump(plotly_fig,open('assets/fore_pp6_2','wb'))
#Evaluate errors
MAE_GB=metrics.mean_absolute_error(y_test,y_pred_GB) 
MSE_GB=metrics.mean_squared_error(y_test,y_pred_GB)  
RMSE_GB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_GB))
cvRMSE_GB=RMSE_GB/np.mean(y_test)
print(MAE_GB,MSE_GB,RMSE_GB,cvRMSE_GB)



#Extreme Gradient Boosting
params = {'n_estimators': 500, 'max_depth': 4, 'min_samples_split': 2,
          'learning_rate': 0.01, 'loss': 'ls'}
GB_model = GradientBoostingRegressor(**params)
XGB_model = XGBRegressor()
XGB_model.fit(X_train, y_train)
y_pred_XGB =XGB_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_XGB[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_XGB[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp7_1','wb'))
fig=plt.scatter(y_test,y_pred_XGB)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_XGB)
pkl.dump(plotly_fig,open('assets/fore_pp7_2','wb'))
#Evaluate errors
MAE_XGB=metrics.mean_absolute_error(y_test,y_pred_XGB) 
MSE_XGB=metrics.mean_squared_error(y_test,y_pred_XGB)  
RMSE_XGB= np.sqrt(metrics.mean_squared_error(y_test,y_pred_XGB))
cvRMSE_XGB=RMSE_XGB/np.mean(y_test)
print(MAE_XGB,MSE_XGB,RMSE_XGB,cvRMSE_XGB)



#Bootstrapping
BT_model = BaggingRegressor()
BT_model.fit(X_train, y_train)
y_pred_BT =BT_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_BT[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_BT[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp8_1','wb'))
fig=plt.scatter(y_test,y_pred_BT)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_BT)
pkl.dump(plotly_fig,open('assets/fore_pp8_2','wb'))
#Evaluate errors
MAE_BT=metrics.mean_absolute_error(y_test,y_pred_BT) 
MSE_BT=metrics.mean_squared_error(y_test,y_pred_BT)  
RMSE_BT= np.sqrt(metrics.mean_squared_error(y_test,y_pred_BT))
cvRMSE_BT=RMSE_BT/np.mean(y_test)
print(MAE_BT,MSE_BT,RMSE_BT,cvRMSE_BT)


#Neural Networks
NN_model = MLPRegressor(hidden_layer_sizes=(10,10,10,10))
NN_model.fit(X_train,y_train)
y_pred_NN = NN_model.predict(X_test)
fig=plt.plot(y_test[1:200])
plt.plot(y_pred_NN[1:200])
plt.show()
plotly_fig=go.Figure()
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_test[1:200],mode='lines',name='actual'))
plotly_fig.add_trace(go.Scatter(x=x_plot, y=y_pred_NN[1:200],mode='lines',name='predition'))
pkl.dump(plotly_fig,open('assets/fore_pp9_1','wb'))
fig=plt.scatter(y_test,y_pred_NN)
plt.show()
plotly_fig=px.scatter(x=y_test, y=y_pred_NN)
pkl.dump(plotly_fig,open('assets/fore_pp9_2','wb'))
#Evaluate errors
MAE_NN=metrics.mean_absolute_error(y_test,y_pred_NN) 
MSE_NN=metrics.mean_squared_error(y_test,y_pred_NN)  
RMSE_NN= np.sqrt(metrics.mean_squared_error(y_test,y_pred_NN))
cvRMSE_NN=RMSE_NN/np.mean(y_test)
print(MAE_NN,MSE_NN,RMSE_NN,cvRMSE_NN)

data.to_csv('data_dash.csv', index=True)

#Prediction didn't really go well, I guess it could still be worse
#Best method for consumption and generation was random forest or uniformised data
#Best method for price was extreme gradient boosting or uniformised data