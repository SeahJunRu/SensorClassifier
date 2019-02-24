import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

#Reading in data from csv
sensors_screen1 = pd.read_csv('data/test_data1536128609.98_screen1.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('Screen 1 Data Shape',sensors_screen1.shape)
sensors_screen2 = pd.read_csv('data/test_data1536128871.84_screen2.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('Screen 2 Data Shape',sensors_screen2.shape)
sensors_device = pd.read_csv('data/test_data1536129261.77_device253.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('Temperature Sensor Data Shape',sensors_device.shape)
sensors_all = pd.read_csv('data/test_data1536129670.55_main.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('All Devices Data Shape',sensors_all.shape)

#Appending expected results to input data
sensors_screen1['PC screen1 state']='1'
sensors_screen1['PC screen2 state']='0'
sensors_screen1['Temperature sensor state']='0'

sensors_screen2['PC screen1 state']='0'
sensors_screen2['PC screen2 state']='1'
sensors_screen2['Temperature sensor state']='0'

sensors_device['PC screen1 state']='0'
sensors_device['PC screen2 state']='0'
sensors_device['Temperature sensor state']='1'

sensors_all['PC screen1 state']='1'
sensors_all['PC screen2 state']='1'
sensors_all['Temperature sensor state']='1'

#Manipulating to get 'X' training data
sensors_screen1_X = sensors_screen1.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])
sensors_screen2_X = sensors_screen2.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])
sensors_device_X = sensors_device.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])
sensors_all_X = sensors_all.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])

#Manipulating to get 'Y' training data
sensors_screen1_Y = sensors_screen1.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])
sensors_screen2_Y = sensors_screen2.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])
sensors_device_Y = sensors_device.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])
sensors_all_Y = sensors_all.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])

#Union data 
train_X = pd.concat([sensors_screen1_X,sensors_screen2_X,sensors_device_X,sensors_all_X], ignore_index=True)
train_Y = pd.concat([sensors_screen1_Y,sensors_screen2_Y,sensors_device_Y,sensors_all_Y], ignore_index=True)

#Reading input data from csv for prediction
P = pd.read_csv('data/test_data.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
#setting default output data of -1
P['PC screen1 state']=-1
P['PC screen2 state']=-1
P['Temperature sensor state']=-1

#K Neighbors Classifier
knn = KNeighborsClassifier()
knn.fit(train_X,train_Y)

#declaring empty pd df
output_df = pd.DataFrame(columns=['Timestamp', 'Maximum current', 'Effective current','PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])

#function to add row to df 
def add_row(df, row):
    df.loc[-1] = row
    df.index = df.index + 1  
    return df.sort_index()

#looping through input for prediction and getting prediction result	
for row in P.itertuples():
  tempP = [[row.Timestamp,row._2,row._3]]
  result = (knn.predict(tempP))
  add_row(output_df, [row.Timestamp,row._2,row._3,result[0,0],result[0,1],result[0,2]]) 

output_df.to_csv('data/test_results.csv',index=False)  
print(output_df)  

