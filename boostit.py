import sys
import math
import numpy as np

def readFile(theFile):
    theList = []
    weights = []
    with open(theFile) as f:
        firstLine = f.readline().strip().split()
        dimension = firstLine[1]   #Second num in line 1 is dimensionality
        numPoints = firstLine[0]   #First num in line 1 is number of points
        
        #Add the rest of the data to data[]
        for line in f.readlines():
            #Turn each point into a list of numbers
            theList.append([float(num) for num in line.split()])
            #Append the weight associated with the point to the end of the list
            weights.append(1.0/float(numPoints))

    return theList, weights

def computeClass(data, middle, boundary):
    computed = (middle.dot([float(i) for i in data])) - boundary
    if(computed > 0):   #Positive
        return 1
    else:               #Negative
        return 0


def sumWeights(theList):
    theSum = 0.0
    for num in theList:
        theSum += num
    return theSum


def sumColumns(theList):
    dimension = len(theList[0])
    colSum = 0
    finalSum = []
    for i in range(dimension):
        for j in range(len(theList)):
            colSum += theList[j][i]
        finalSum.append(colSum)
        colSum = 0
    return finalSum
            
def computeCentroid(trainPos, trainNeg, weightsNeg, weightsPos):
    sumWeightsNeg = 1.0/sumWeights(weightsNeg)
    sumWeightsPos = 1.0/sumWeights(weightsPos)
    
    #Multiply points by their weights
    for i in range(len(trainPos)):
        trainPos[i] = [num * weightsPos[i] for num in trainPos[i]]
        
    for i in range(len(trainNeg)):
        trainNeg[i] = [num * weightsNeg[i] for num in trainNeg[i]]
    
    #Centroid of Positives
    posCentroid = np.mean(trainPos[:], 0)#sumColumns(trainPos) #np.mean(trainPos[0:-1], 0)
    #posCentroid = [num * len(trainPos) for num in posCentroid]
    posCentroid = [num * sumWeightsPos for num in posCentroid]
    posCentroid = np.asarray(posCentroid)
    print(posCentroid)
    #Centroid of Negatives
    negCentroid = np.mean(trainNeg[:], 0)#sumColumns(trainNeg) #np.mean(trainNeg[0:-1], 0)
    #negCentroid = [num * len(trainNeg) for num in negCentroid]
    negCentroid = [num * sumWeightsNeg for num in negCentroid]
    negCentroid = np.asarray(negCentroid)
    print(negCentroid)
    return posCentroid,negCentroid


def boost(trainPos, trainNeg, posCentroid, negCentroid, middle, boundary):
    incorrectP = [] #Misclassified points in trainPos
    incorrectN = [] #Misclassified points in trainNeg

    for i in range(0, len(trainPos)):
        result = computeClass(trainPos[i], middle, boundary)
        if(result == 0):  #False Negative
            incorrectP.append(i)  #Append index of the misclassified point
        
    for i in range(0, len(trainNeg)):
        result = computeClass(trainNeg[i], middle, boundary)
        if(result == 1):  #False Positive
            incorrectN.append(i)  #Append index of the misclassified point
    print(incorrectP)
    print(incorrectN)
    err = (float(len(incorrectP) + len(incorrectN)) / float(len(trainPos) + len(trainNeg)))
    alpha = 0.5 * (math.log((1.0 - err) / err))
    increase = (2.0 * (1 - err))
    decrease = (2.0 * err)

    printResults(err, alpha, increase, decrease)

    for i in range(len(trainPos)):
        if i in incorrectP:
            weightsPos[i] = weightsPos[i]/increase
        else:
            weightsPos[i] = weightsPos[i]/decrease

    for i in range(len(trainNeg)):
        if i in incorrectN:
            weightsNeg[i] = weightsNeg[i]/increase
        else:
            weightsNeg[i] = weightsNeg[i]/decrease
    
    #Calculate new centroids
    posCentroid, negCentroid = computeCentroid(trainPos, trainNeg, weightsPos, weightsNeg)
    #Calculate new middle
    middle = posCentroid - negCentroid
    #Calculate new boundary
    boundary = (((posCentroid + negCentroid).T).dot(middle))/2.0


def printResults(err, alpha, increase, decrease):
    sys.stdout.write("Error = ")
    print("%.2f" % err)
    sys.stdout.write("Alpha = ")
    print("%.4f" % alpha)
    sys.stdout.write("Factor to increase weights = ")
    print("%.4f" % increase)
    sys.stdout.write("Factor to decrease weights = ")
    print("%.4f" % decrease)
    

T = sys.argv[1]
train_pos = sys.argv[2] #Training positives
train_neg = sys.argv[3] #Training negatives
test_pos = sys.argv[4] #Testing positives
test_neg = sys.argv[5] #Testing negatives

trainPos, weightsPos = readFile(train_pos)
trainNeg, weightsNeg = readFile(train_neg)
testPos = readFile(test_pos)
testNeg = readFile(test_neg)

posCentroid, negCentroid = computeCentroid(trainPos, trainNeg, weightsPos, weightsNeg)

middle = posCentroid - negCentroid
boundary = (((posCentroid + negCentroid).T).dot(middle))/2.0
print(boundary)

counter = 1
for i in range(int(T)):
    sys.stdout.write("Iteration ")
    sys.stdout.write(str(counter))
    print(":")
    boost(trainPos, trainNeg, posCentroid, negCentroid, middle, boundary)
    counter += 1
    
FP = 0
FN = 0
errPos = 0.0
errNeg = 0.0
    
with open(test_pos) as f:
    data = f.readline().split()
    numPoints = data[0]
    dim = data[1]
    for line in f.readlines():
        point = [float(num) for num in line.split()]
        if(point):
            guess = computeClass(point, middle, boundary)
            if(guess == 0):
                FN += 1
    errPos = float(FN)/float(numPoints)

with open(test_neg) as f:
    data = f.readline().split()
    numPoints =data[0]
    dim = data[1]
    for line in f.readlines():
        point =[float(num) for num in line.split()]
        if(point):
            guess = computeClass(point, middle, boundary)
            if(guess == 1):
                FP += 1
    errNeg = float(FP)/float(numPoints)
        
totalErr = str((errPos + errNeg)/2.0)
pctErr = totalErr + "%"
    
print("Testing:")
sys.stdout.write("False positives: ")
print(FP)
sys.stdout.write("False negatives: ")
print(FN)
sys.stdout.write("Error rate: ")
print(pctErr)
