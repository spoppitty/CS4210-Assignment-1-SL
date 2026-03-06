#-------------------------------------------------------------------------
# AUTHOR: Sarah Liu
# FILENAME: knn.py
# SPECIFICATION: Compute the LOO-CV error rate for a 1NN classifier
# FOR: CS 4210- Assignment #2
# TIME SPENT: 1 hour
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd

#Reading the data in a csv file using pandas
db = []
df = pd.read_csv('email_classification.csv')
for _, row in df.iterrows():
    db.append(row.tolist())

# keep track of errors for error rate 
num_errors = 0

#Loop your data to allow each instance to be your test set
for i in db:

    #Add the training features to the 20D array X removing the instance that will be used for testing in this iteration.
    #For instance, X = [[1, 2, 3, 4, 5, ..., 20]].
    #Convert each feature value to float to avoid warning messages
    X = []
    for j in db:
        if j != i:
            X.append([float(v) for v in j[0:20]])

    #Transform the original training classes to numbers and add them to the vector Y.
    #Do not forget to remove the instance that will be used for testing in this iteration.
    #For instance, Y = [1, 2, ,...].
    #Convert each feature value to float to avoid warning messages
    label_map = {'ham': 0, 'spam': 1}
    Y = []
    for j in db:
        if j != i:
            Y.append(label_map[j[20]])

    #Store the test sample of this iteration in the vector testSample
    testSample = [float(v) for v in i[0:20]]
    true_label = label_map[i[20]]

    #Fitting the knn to the data using k = 1 and Euclidean distance (L2 norm)
    clf = KNeighborsClassifier(n_neighbors=1, metric='euclidean')
    clf.fit(X, Y)

    #Use your test sample in this iteration to make the class prediction. For instance:
    #class_predicted = clf.predict([[1, 2, 3, 4, 5, ..., 20]])[0]
    class_predicted = clf.predict([testSample])[0]

    #Compare the prediction with the true label of the test instance to start calculating the error rate.
    if class_predicted != true_label:
        num_errors += 1

#Print the error rate
error_rate = num_errors / len(db)
print(f"Error Rate: {error_rate}")






