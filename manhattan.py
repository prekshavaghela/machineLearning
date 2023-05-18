#part b4: manhattan distances KNN: 
import math
import matplotlib.pyplot as plt
import csv


def knnManhattan(fileTrain, fileTest, k ):
    
    #open the Csv file for testing data 
    testing_data = csv.reader(fileTest)
    
    #making one array each property but each row corresponds to the same index in each of these arrays
    testAge = []
    testYear = []
    testNodes = []
    testClass = []

    #add all test data properties into a variable (or dev set in hyperparameter tuning)
    for test in testing_data:
        testAge.append( int(test[0]) )
        testYear.append( int(test[1]) )
        testNodes.append( int(test[2]) )
        testClass.append( int(test[3]) )

    lenTest = len(testAge)

    #open the Csv file for training data 
    training_data = csv.reader(fileTrain)
    
     #making one array each property but each row corresponds to the same index in each of these arrays
    trainAge = []
    trainYear = []
    trainNodes = []
    trainClass = []

    #add all training data properties into appropriate array
    #i[0] is age. i[1] is year. i[2] is number nodes. i[3] is classification
    for train in training_data:
        trainAge.append( int(train[0]) )
        trainYear.append( int(train[1]) )
        trainNodes.append( int(train[2]) )
        trainClass.append( int(train[3]) )
    
    lenTrain = len(trainAge)

    KnnClass = []
    NumCorrectClass1 = 0
    NumCorrectClass2 = 0
    Acc = 0
    BAcc = 0

    for testIndex in range(lenTest):
        #get the test point and store each test points computed distance
        distanceAllTraining = []
        #for all training data compute the distance between them and test sample
        for i in range(lenTrain): 
            #manhattan distance
            dist =  abs(trainAge[i]-testAge[testIndex]) + abs(trainYear[i]-testYear[testIndex])+ abs(trainNodes[i] - testNodes[testIndex]) 
            distanceAllTraining.append((dist, trainClass[i]))

        countClass1 = 0
        countClass2 = 0
        
        #the column in excel -1. check that distance is being computed correctly
        #print(distanceAllTraining[48], trainAge[48], trainYear[48])
        distanceAllTraining.sort()
        
        #get the k minimum distances between and then see which class that they are in
        for i in range(0,k):
            distance, classDist = distanceAllTraining[i]
            if classDist == 1:
                countClass1+= 1
            else: 
                countClass2 +=1

        # the Knnclass is whichever one is max
        if countClass1 > countClass2:
            KnnClass.append(1)
        else:
            KnnClass.append(2)  

    # to get the number correct compare the class u got for each test sample vs actual
    n= len(KnnClass)
    for i in range(n):
        if( KnnClass[i] == testClass[i]):
            if( testClass[i] == 1):
                NumCorrectClass1 +=1
            else:
                NumCorrectClass2 +=1
    
    #and get the actual number of each to compute BAcc 
    actualCount1 = 0
    actualCount2 = 0
    for val in testClass:
        if val == 1:
            actualCount1 +=1
        else:
            actualCount2 +=1
    
    #compute accuracy
    Acc = (NumCorrectClass1+ NumCorrectClass2)/(n)
    BAcc = (0.5* NumCorrectClass1)/actualCount1 + (0.5*NumCorrectClass2)/actualCount2

    #close the files 
    fileTest.close()
    fileTrain.close()
    
    #return the classifications for each test sample and the overall accuracies of this K value
    return KnnClass, Acc, BAcc

#hyperparameter tuning for all the k values specified 

kValuesManhattan = [1,3,5,7] #set k value.
classificationManhattan = []
accManhattan = []
baccManhattan = []
#for every k value: get the KNN using manhattan distance
for i in range(len(kValuesManhattan)):
    # print(kValues[i])
    fileTrain = open('data_train.csv')
    fileDev = open('data_dev.csv') #file closes in function.
    classCurr, accCurr, baccCurr = knnManhattan(fileTrain, fileDev, kValuesManhattan[i])
    classificationManhattan.append(classCurr)
    accManhattan.append(accCurr)
    baccManhattan.append(baccCurr)
    fileTrain.close()
    fileDev.close()

print(accManhattan)
print(baccManhattan)
#plot of the hyperparameter tuning with manhattan 
plt.scatter( kValuesManhattan, accManhattan, label="ACC ")
plt.scatter( kValuesManhattan, baccManhattan, label="BACC")
plt.title("ACC And BACC based on the k-values Manhattan")
plt.legend()
plt.xlabel("K-values")
plt.ylabel("Accuracy ")
plt.show()

fileTrain = open('data_train.csv')
fileTest = open('data_test.csv')
classTest, accTest, baccTest = knnManhattan(fileTrain, fileTest, 5)
print(3, " : ", classTest, accTest, baccTest)
fileTrain.close()
fileTest.close()

#k**
fileTrain = open('data_train.csv')
fileTest = open('data_test.csv')
classTest, accTest, baccTest = knnManhattan(fileTrain, fileTest, 7)
print(3, " : ", classTest, accTest, baccTest)
fileTrain.close()
fileTest.close()

