import csv
import math
import matplotlib.pyplot as plt

def knn(fileTrain, fileTest, k ):
    
    #open the Csv file for testing data 
    testing_data = csv.reader(fileTest)
    
    #making one array each property but each row corresponds to the same index in each of these arrays
    testAge = []
    testYear = []
    testNodes = []
    testClass = []

    #add all test data properties into a variable
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

    #to compute the accuracy metrics we need the overall counts and the acurracy count
    KnnClass = []
    NumCorrectClass1 = 0
    NumCorrectClass2 = 0
    Acc = 0
    BAcc = 0

    for testIndex in range(lenTest):
        #get the test point and store each test points computed distance
        distanceAllTraining = []
        #for all training data compute the euclidean distance between them and test sample
        for i in range(lenTrain): 
            dist = math.sqrt( (trainAge[i]-testAge[testIndex])**2 + (trainYear[i]-testYear[testIndex])**2 + (trainNodes[i] - testNodes[testIndex])**2 )
            distanceAllTraining.append((dist, trainClass[i]))

        countClass1 = 0
        countClass2 = 0
        
        #sort by ascending order so k minimums are the first k elements
        distanceAllTraining.sort()

        for i in range(0,k):
            #runs k times
            distance, classDist = distanceAllTraining[i]
            if classDist == 1:
                countClass1+= 1
            else: 
                countClass2 +=1

        # print(countClass1, countClass2)
        if countClass1 > countClass2:
            KnnClass.append(1) #if most of the K samples were class 1
        else:
            KnnClass.append(2)  # if most of the k samples were class 2 

    # print( len(KnnClass) == len(testClass))
    # check if the predicted and actual values for class are the same 
    n= len(KnnClass)
    for i in range(n):
        if( KnnClass[i] == testClass[i]):
            if( testClass[i] == 1):
                NumCorrectClass1 +=1
            else:
                NumCorrectClass2 +=1
    
    actualCount1 = 0
    actualCount2 = 0
    #get a total running count of classA and classB actual 
    for val in testClass:
        if val == 1:
            actualCount1 +=1
        else:
            actualCount2 +=1
    
    #compute and return accuracies 
    Acc = (NumCorrectClass1+ NumCorrectClass2)/(n)
    BAcc = (0.5* NumCorrectClass1)/actualCount1 + (0.5*NumCorrectClass2)/actualCount2

    fileTest.close()
    fileTrain.close()
    
    return KnnClass, Acc, BAcc


#b1: running a simple KNN
k= 3 
fileTrain = open('data_train.csv')
fileTest = open('data_test.csv')
classTest, accTest, baccTest = knn(fileTrain, fileTest, k)
print( classTest, accTest, baccTest)
fileTrain.close()
fileTest.close()


# b2. the hyperparamater tuning 
kValues = [1,3,5,7,9,11] #set k value.
classification = []
acc = []
bacc = []
for i in range(len(kValues)):
    # print(kValues[i])
    fileTrain = open('data_train.csv')
    fileDev = open('data_dev.csv')
    classCurr, accCurr, baccCurr = knn(fileTrain, fileDev, kValues[i])
    classification.append(classCurr)
    acc.append(accCurr)
    bacc.append(baccCurr)
    fileTrain.close()
    fileDev.close()


print(acc)
print(bacc)
#plot of the hyperparameter tuning 
plt.scatter( kValues, acc, label="ACC ")
plt.scatter( kValues, bacc, label="BACC")
plt.title("ACC And BACC based on the k-values")
plt.legend()
plt.xlabel("K-vaues")
plt.ylabel("Accuracy ")
plt.show()

#compute the best k-value with max acccuracy. will pick lowest one with best accuracy
maxValueAcc = max(acc)
maxValueBacc = max(bacc)
kACC = kValues[acc.index(maxValueAcc)]
kBACC= kValues[bacc.index(maxValueBacc)]

print( "K*  =  ",  kACC)
print( "K** =  ", kBACC)

#part b3: try to run knn on the test set with the results of the hyperparameter tuning
#k*
fileTrain = open('data_train.csv')
fileTest = open('data_test.csv')
classTest, accTest, baccTest = knn(fileTrain, fileTest, kACC)
print(kACC, " : ", classTest, accTest, baccTest)
fileTrain.close()
fileTest.close()

#k**
fileTrain = open('data_train.csv')
fileTest = open('data_test.csv')
classTest, accTest, baccTest = knn(fileTrain, fileTest, kBACC)
print(kBACC, " : ", classTest, accTest, baccTest)
fileTrain.close()
fileTest.close()


