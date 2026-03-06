#-------------------------------------------------------------------------
# AUTHOR: Sarah Liu
# FILENAME: decision_tree.py
# SPECIFICATION: Derive a depth-2 decision tree produced by the ID3 algorithm by alphabetical order.
# FOR: CS 4210- Assignment #1
# TIME SPENT: about 1 hour, 30 minutes
#-----------------------------------------------------------*/

#IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard
# dictionaries, lists, and arrays

#importing some Python libraries
from sklearn import tree
import matplotlib.pyplot as plt
import csv
db = []
X = []
Y = []

#reading the data in a csv file
with open('contact_lens.csv', 'r') as csvfile:
  reader = csv.reader(csvfile)
  for i, row in enumerate(reader):
      if i > 0: #skipping the header
         db.append (row)

#encode the original categorical training features into numbers and add to the 4D array X.
numFeatures = len(db[0]) - 1    # number of features/classes
featureValues = []
for col in range(numFeatures): 
    values = sorted(set(row[col].strip() for row in db))    # sorts alphabetically
    featureValues.append(values)

featureMaps = []
for col in range(numFeatures):
    mapping = {}
    for index, val in enumerate(featureValues[col]):
        mapping[val] = index
    featureMaps.append(mapping)

X = []
for row in db:
    everyRow = []
    for i in range(numFeatures):
        everyRow.append(featureMaps[i][row[i].strip()])
    X.append(everyRow)

#encode the original categorical training classes into numbers and add to the vector Y.
classValues = sorted(set(row[numFeatures].strip() for row in db))
classMap = {}
for index, val in enumerate(classValues):
    classMap[val] = index

Y = []
for row in db:
    Y.append(classMap[row[numFeatures].strip()])

#fitting the depth-2 decision tree to the data using entropy as your impurity measure
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, random_state=0)
clf = clf.fit(X, Y)

#plotting decision tree
tree.plot_tree(clf, feature_names=['Age', 'Spectacle', 'Astigmatism', 'Tear'], class_names=['Yes','No'], filled=True, rounded=True)
plt.show()