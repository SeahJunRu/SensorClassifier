import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier

sensors_screen1 = pd.read_csv('data/test_data1536128609.98_screen1.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('screen1',sensors_screen1.shape)
sensors_screen2 = pd.read_csv('data/test_data1536128871.84_screen2.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('screen2',sensors_screen2.shape)
sensors_device = pd.read_csv('data/test_data1536129261.77_device253.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('device',sensors_device.shape)
sensors_all = pd.read_csv('data/test_data1536129670.55_main.csv', names=['Timestamp', 'Maximum current', 'Effective current'])
print('all',sensors_all.shape)

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

sensors_screen1_X = sensors_screen1.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])
sensors_screen2_X = sensors_screen2.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])
sensors_device_X = sensors_device.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])
sensors_all_X = sensors_all.drop(columns=['PC screen1 state', 'PC screen2 state', 'Temperature sensor state'])

print(sensors_screen1_X.head())

sensors_screen1_Y = sensors_screen1.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])
sensors_screen2_Y = sensors_screen2.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])
sensors_device_Y = sensors_device.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])
sensors_all_Y = sensors_all.drop(columns=['Timestamp', 'Maximum current', 'Effective current'])

print(sensors_screen1_Y.head())

train_X = pd.concat([sensors_screen1_X,sensors_screen2_X,sensors_device_X,sensors_all_X], ignore_index=True)

train_Y = pd.concat([sensors_screen1_Y,sensors_screen2_Y,sensors_device_Y,sensors_all_Y], ignore_index=True)

P = [[1536129671.00052,24.41,17.3]]

#{Decision Tree Model}
clf = DecisionTreeClassifier()
clf = clf.fit(train_X,train_Y)
print ("\n1) Using Decision Tree Prediction is " + str(clf.predict(P)))

#{K Neighbors Classifier}
knn = KNeighborsClassifier()
knn.fit(train_X,train_Y)
print ("2) Using K Neighbors Classifier Prediction is " + str(knn.predict(P)))

#{using MLPClassifier}
rfor = RandomForestClassifier()
rfor.fit(train_X,train_Y)
print ("3) Using RandomForestClassifier Prediction is " + str(rfor.predict(P)) +"\n")



