#-------------------------------------------------------------------------
# AUTHOR: Sarah Liu
# FILENAME: naive_bayes.py
# SPECIFICATION: Read and output the classification of each of the 10 instances 
# FOR: CS 4210- Assignment #2
# TIME SPENT: about 1 hour 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd

dbTraining = []
dbTest = []

#Reading the training data using Pandas
df = pd.read_csv('weather_training.csv')
for _, row in df.iterrows():
    dbTraining.append(row.tolist())

#Transform the original training features to numbers and add them to the 4D array X.
#For instance Sunny = 1, Overcast = 2, Rain = 3, X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
outlook_map = {'Sunny': 1, 'Overcast': 2, 'Rain': 3}
temperature_map = {'Hot': 1, 'Mild': 2, 'Cool': 3}
humidity_map = {'High': 1, 'Normal': 2}
wind_map = {'Weak': 1, 'Strong': 2}

X = []
for instance in dbTraining:
    X.append([
        outlook_map[instance[1]],
        temperature_map[instance[2]],
        humidity_map[instance[3]],
        wind_map[instance[4]]
    ])

#Transform the original training classes to numbers and add them to the vector Y.
#For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
Y = []
for instance in dbTraining:
    if instance[5] == 'Yes':
        Y.append(1)
    else:
        Y.append(2)

#Fitting the naive bayes to the data using smoothing
clf = GaussianNB(var_smoothing=1e-9)
clf.fit(X, Y)

#Reading the test data using Pandas
df = pd.read_csv('weather_test.csv')
for _, row in df.iterrows():
    dbTest.append(row.tolist())

#Printing the header os the solution
print(f"{'Day':<8}{'Outlook':<10}{'Temperature':<13}{'Humidity':<10}{'Wind':<8}{'PlayTennis':<11}{'Confidence'}")

#Use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
for instance in dbTest:
    sample = [[
        outlook_map[instance[1]],
        temperature_map[instance[2]],
        humidity_map[instance[3]],
        wind_map[instance[4]]
    ]]

    probs = clf.predict_proba(sample)[0]
    pred_class = clf.predict(sample)[0]

    class_index = list(clf.classes_).index(pred_class)
    confidence = probs[class_index]
    
    # print if the confidence is >= 0.75
    if confidence >= 0.75:
        pred_label = 'Yes' if pred_class == 1 else 'No'
        print(
            f"{str(instance[0]):<8}"
            f"{instance[1]:<10}"
            f"{instance[2]:<13}"
            f"{instance[3]:<10}"
            f"{instance[4]:<8}"
            f"{pred_label:<11}"
            f"{confidence:.2f}"
        )


