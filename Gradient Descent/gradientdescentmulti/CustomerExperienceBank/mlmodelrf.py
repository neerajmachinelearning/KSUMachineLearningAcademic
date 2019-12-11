# Importing the libraries

import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier



# Importing the dataset
def loaddata():
    dataset = pd.read_csv(r'C:\Users\370313\Desktop\Hackathon\finaldataset.csv')
    X = dataset.iloc[:, [0, 1, 2, 3, 4]].values
    y = dataset.iloc[:, 5].values
    print(X)
    return X, y

# pre process the data (Split it in to training set and test set
def datapreprocessing(X, y):
    labelencoder = LabelEncoder()
    y = labelencoder.fit_transform(y)

    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

    print(f"Total number of data is {len(X)}")
    print(f"Number of training data is {len(X_train)}")
    print(f"Number of testing data is {len(X_test)}")


    # Feature Scalling for the X_train - This code is commented right now because in our current data set we do not need
    # feature scalling. if we can use this in future if needed.

    # sc_X = StandardScaler()
    # X_train = sc_X.fit_transform(X_train)
    # X_test = sc_X.transform(X_test)

    return X_train, X_test, y_train, y_test


# create confusion matrix - providers how many test resuls got from test set or correct or incorrect.
def confustionmatrix(y_test, y_pred):
    # Making confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"Descision Matrix is: {cm}")


# Finding accuracy of the model
def getaccuracy(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print(f"accuracy of the model is: {accuracy}")
    return accuracy

# gives the result of the new input provided by the api
def getresults(X_new, classifier):
    sc_X = StandardScaler()
    # X_new = sc_X.fit_transform(X_new)
    # print(f"input given after categorization is {X_new}")
    y_new = classifier.predict(X_new)
    print(f"prediction for given input is {y_new}")
    status = y_new[0]
    return status



def main(X_new):
    # X_new = [200, 160, 150, 68, 0.5]
    X, y = loaddata()
    X_train, X_test, y_train, y_test = datapreprocessing(X, y)

    # Fitting Random Forest Classification to the Training set
    classifier = RandomForestClassifier(n_estimators=9, criterion='entropy', random_state=0)
    classifier.fit(X_train, y_train)
    print("Actual values of test set is " )
    print(y_test)
    # predicting the Test set results
    y_pred = classifier.predict(X_test)
    print("predicted value of test set is ")
    print(y_pred)
    accuracy = getaccuracy(y_test, y_pred)
    confustionmatrix(y_test, y_pred)
    X_new = [X_new]
    status = getresults(X_new, classifier)

    if status == 0:
        provide = "cabin"
    else:
        provide = "checkin"
    print(f"Cabin or forced checkin? : {provide}")
    return accuracy, provide

