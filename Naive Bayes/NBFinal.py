import random
import csv
import math
import operator
import itertools as it
import pandas as pd
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


def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    print(separated)
    return separated


def mean(numbers):
    number1 = tuple(map(float, numbers))
    # a = math.fsum(numbers)
    print("======mean=====calculated======")
    print(sum(number1)/float(len(number1)))
    return sum(number1)/float(len(number1))

def stdev(numbers):
    print("length of the numbers")
    print(len(numbers))
    avg = mean(numbers)
    print(avg)
    power = 0
    for x in numbers:
        # print("print value of x and average")
        # print(x)
        # print(avg)
        power = pow((x-avg), 2)

        # print("print value of power")
        # print(power)
        s = power + 0
    print("======value of s======")
    print(s)
    variance = s /float(len(numbers))

    #variance =float(sum([pow(x - avg, 2) for x in numbers])/float(len(numbers)))
    print("======variance=========")
    print(variance)
    # print(math.sqrt(variance))
    # print("=======numbers length=======")
    # print(float(len(numbers)))
    return math.sqrt(variance)

def summarize(dataset):
    df = pd.DataFrame(data=dataset)
    data = df.iloc[:, 0:4]
    col = df.head(4)
    for i in col:
        summaries = [mean(data[i]), stdev(data[i])]

    #print(mean(data[0]))
    print("============summaries===============")
    print(summaries)
    #del summaries[-1]
    return summaries


def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    # print("==================Separated===================")
    # print(separated)
    summaries = {}
    for classValue, instances in separated.items():
        # print(classValue)
        # print(instances)
        summaries[classValue] = summarize(instances)
    # print("==================Summary===================")
    # print(summaries)
    return summaries


def calculateProbability(x, mean, stdev):
    # print("x" + "mean" + "stdev")
    # print(x, mean, stdev)
    if stdev != 0.0:
        exponent = math.exp(-(math.pow(mean,2)/(2*math.pow(stdev,2))))
        # print("exponent")
        return (1 / (math.sqrt(2*math.pi) * stdev)) * exponent
    else:
        return 0



def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 1
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] = probabilities[classValue] * calculateProbability(x, mean, stdev)
    return probabilities


def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    print("============probabilities=======================")
    print(probabilities)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel


def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    print("===================================predition===================")
    print(predictions)
    return predictions


def getAccuracy(testSet, predictions):
    correct = 0
    for i in range(len(testSet)):
        if testSet[i][-1] == predictions[i]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0


def nbmain():
    #Initializing different datasets
    train_dataset = []
    test_dataset = []
    split = 0.60
    #call the function load_dataset and provide above information including file name.
    load_dataset(r'C:\Neeraj Sharma\Kennesaw State\Fall 2019\Machine Learning\MLProjects\Fisher_iris_data.csv', split, train_dataset, test_dataset)
    print(len(train_dataset))
    print("=================================================================")
    print(len(test_dataset))
    summaries = summarizeByClass(train_dataset)
    predictions = getPredictions(summaries, test_dataset)
    # print(predictions)
    accuracy = getAccuracy(test_dataset, predictions)
    print(f"Accuracy is : {accuracy}%")

nbmain()

