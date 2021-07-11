import pandas as pd

weather=pd.read_csv('weather.csv')

#print(weather.info())
#print(weather)

weather['time']=pd.to_datetime(weather['time'])
weather=weather.set_index('time', drop=True)

#print(weather.info())
#print(weather)

#weather_main is essentially the same as weather_id
#temp_min and temp_max are the same as temp
weather=weather.drop(columns=['weather_main', 'temp_min', 'temp_max'])

#Separating by city
weather_madrid=weather[weather['city_name'] == 'Madrid']
weather_valencia=weather[weather['city_name'] == 'Valencia']
weather_seville=weather[weather['city_name'] == 'Seville']
weather_barcelona=weather[weather['city_name'] == 'Barcelona']
weather_bilbao=weather[weather['city_name'] == 'Bilbao']


#Grouping weather by mean for the 5 cities. considering mean as values for the country
weather=pd.concat([weather_madrid, weather_valencia, weather_seville, weather_barcelona, weather_bilbao]).groupby(level=0).mean()

#Weather file has temperature, wind speed and direction, pressure, humidity, rain, snow, clouds and a weather id.
#I will add solar radiation from a hourly average for Spain

print(weather.info())
print(weather)

weather.to_csv('weather_final.csv', index=True)