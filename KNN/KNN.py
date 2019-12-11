import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import csv
import math
import operator
import tkinter.messagebox
from tkinter import *

#load data set and split it to training set and data set
def load_dataset(filename, split, train_dataset=[], test_dataset=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        print(f"Total Data: {len(dataset)-1}")
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x+1][y] = dataset[x+1][y]
            if random.random() < split:    #Return the next random floating point number in the range [0.0, 1.0).
                train_dataset.append(dataset[x+1])
            else:
                test_dataset.append(dataset[x+1])


# find Euclidean distance between 2 data points
def calculate_euclidean_distance(instance1, instance2, length):
    dist_btw_points = 0
    for x in range(length):
        dist_btw_points += pow((float(instance1[x]) - float(instance2[x])), 2)
    return math.sqrt(dist_btw_points)


# finding the neighbors of the test_instance after sorting them by distance
def fetch_neighbors(train_dataset, test_instance, k):

    dist = []
    length = len(test_instance) - 1
    for x in range(len(train_dataset)):
        dist_btw_points = calculate_euclidean_distance(test_instance, train_dataset[x], length)
        dist.append((train_dataset[x], dist_btw_points))
    dist.sort(key=operator.itemgetter(1))
    neighbors_list = []
    for x in range(k):
        neighbors_list.append(dist[x][0])
    return neighbors_list


# Calculate the voting for a perticular result and fetch the predicted and actual value
def fetch_response(neighbors_list):
    class_votes = {}
    for x in range(len(neighbors_list)):
        response = neighbors_list[x][-1]
        if response in class_votes:
            class_votes[response] += 1
        else:
            class_votes[response] = 1
    sorted_votes = sorted(class_votes.items(), key=operator.itemgetter(1), reverse=True)
    print(f"Sorted votes {sorted_votes}")
    return sorted_votes[0][0]


# Calculating accuracy of the KNN program
def fetch_accuracy(test_dataset, knn_predictions):
    correct_prediction = 0
    for x in range(len(test_dataset)):
        if test_dataset[x][-1] == knn_predictions[x]:
            correct_prediction += 1
    return (correct_prediction / float(len(test_dataset))) * 100.0


# Main program defined here
def knnmain(k):
    # Initializing different datasets
    train_dataset = []
    test_dataset = []
    split = 0.67
    #call the function load_dataset and provide above information including file name.
    load_dataset(r'C:\Neeraj Sharma\Kennesaw State\Fall 2019\Machine Learning\KNN Assignment\irisdataset.csv', split, train_dataset, test_dataset)
# generating knn_predictions
    knn_predictions = []
    #display split dataset and total count
    print(f"Total training data : {len(train_dataset)}")
    print(f"Total number of test data: {len(test_dataset)}")
    #print(train_dataset)
    #print(test_dataset)
   # k = int(input("Please enter the value of k: "))
    for x in range(len(test_dataset)):
        neighbors_list = fetch_neighbors(train_dataset, test_dataset[x], k)
        result = fetch_response(neighbors_list)
        knn_predictions.append(result)
        print('> Predicted= '+ repr(result)+', Actual= '+repr(test_dataset[x][-1]))
    knn_accuracy = fetch_accuracy(test_dataset, knn_predictions)
    print('Accuracy: ' + repr(knn_accuracy) + '%')
    tkinter.messagebox.showinfo("Accuracy is: ",knn_accuracy,)   #displaying accuracy in the message box.

#=======================================GUI Portion=============================================================
fields = 'Enter the value of K',

#This function is used to fetch value entered in the input box and it is passed to the knnmain function
def fetch(entries):
    for entry in entries:
        field = entry[0]
        text  = entry[1].get()
       # print('%s: "%s"' % (field, text))
        my_input_str = text
    #
    # #my_input_str = input('please enter the input string: ')
    if my_input_str == "":
        return root.quit
    else:
        knnmain(int(my_input_str))
    return

#This function creates the form
def makeform(root, fields):
    entries = []
    for field in fields:
        row = Frame(root)
        lab = Label(row, width=15, text=field, anchor='w')
        ent = Entry(row)
        row.pack(side=TOP, fill=X, padx=5, pady=5)
        lab.pack(side=LEFT)
        ent.pack(side=RIGHT, expand=YES, fill=X)
        entries.append((field, ent))

    return entries

if __name__ == '__main__':
    root = Tk()   #create an object of the TK function to create the window.
    root.title("K-Nearest Neighbor") #window title
    ents = makeform(root, fields)    #call makeform function to create the form.
    root.bind('<Return>', (lambda event, e=ents: fetch(e)))
    #Creating button Run KNN in the window
    b1 = Button(root, text='Run KNN',
                command=lambda e=ents: fetch(e))
   #Creatign button Quit in the input window
    b1.pack(side=LEFT, padx=5, pady=5)
    b2 = Button(root, text='Quit', command=root.destroy)
    b2.pack(side=LEFT, padx=5, pady=5)
    #loop the window until user exits from the program
    root.mainloop()
