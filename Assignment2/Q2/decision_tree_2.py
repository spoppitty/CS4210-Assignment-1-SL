#-------------------------------------------------------------------------
# AUTHOR: Sarah Liu
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train and Test 3 models, run 10 times, and print the accuracy
# FOR: CS 4210- Assignment #2
# TIME SPENT: about 1 hour 30 mins
#-----------------------------------------------------------*/

#IMPORTANT NOTE: YOU ARE ALLOWED TO USE ANY PYTHON LIBRARY TO COMPLETE THIS PROGRAM

#Importing some Python libraries
from sklearn import tree
import pandas as pd

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

#Reading the test data in a csv file using pandas
dbTest = []
df_test = pd.read_csv('contact_lens_test.csv')
for _, row in df_test.iterrows():
    dbTest.append(row.tolist())

# categorical numeric mappings 
age_map = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3}
spectacle_map = {'Myope': 1, 'Hypermetrope': 2}
astigmatism_map = {'Yes': 1, 'No': 2}
tear_map = {'Reduced': 1, 'Normal': 2}
class_map = {'Yes': 1, 'No': 2}

for ds in dataSets:

    dbTraining = []
    X = []
    Y = []

    #Reading the training data in a csv file using pandas
    df_train = pd.read_csv(ds)
    for _, row in df_train.iterrows():
        dbTraining.append(row.tolist())

    #Transform the original categorical training features to numbers and add to the 4D array X.
    #For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3, X = [[1, 1, 1, 1], [2, 2, 2, 2], ...]]
    for entry in dbTraining:
        age = age_map[entry[0].strip()]
        spec = spectacle_map[entry[1].strip()]
        ast = astigmatism_map[entry[2].strip()]
        tear = tear_map[entry[3].strip()]

        X.append([age, spec, ast, tear])
    
    #Transform the original categorical training classes to numbers and add to the vector Y.
    #For instance Yes = 1 and No = 2, Y = [1, 1, 2, 2, ...]
    for entry in dbTraining:
        Y.append(class_map[entry[4].strip()])
    
    # keep track of the overall accuracy
    total_accuracy = 0.0

    #Loop your training and test tasks 10 times here
    for i in range (10):

       # fitting the decision tree to the data using entropy as your impurity measure and maximum depth = 5
        clf = tree.DecisionTreeClassifier(
            criterion='entropy',
            max_depth=5
        )
        clf.fit(X, Y)

        correct = 0
        total = 0

       #Read the test data and add this data to dbTest
       # it was already read earlier 

        for data in dbTest:
           #Transform the features of the test instances to numbers following the same strategy done during training,
           #and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           #where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
            age_t = age_map[data[0].strip()]
            spec_t = spectacle_map[data[1].strip()]
            ast_t = astigmatism_map[data[2].strip()]
            tear_t = tear_map[data[3].strip()]

            class_predicted = clf.predict([[age_t, spec_t, ast_t, tear_t]])[0]

           #Compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy.
            true_label = class_map[data[4].strip()]

            if class_predicted == true_label:
                correct += 1
            
            total += 1
        
        # calculate accuracy and update total accuracy
        accuracy = correct / total
        total_accuracy += accuracy

    #Find the average of this model during the 10 runs (training and test set)
    average_accuracy = total_accuracy / 10.0

    #Print the average accuracy of this model during the 10 runs (training and test set).
    #Your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print(f"Average accuracy of {ds}: {average_accuracy}")




