from sklearn.ensemble import RandomForestClassifier 
from sklearn import tree
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
import matplotlib.pyplot as plt
import random
import pandas


## removed the NAN row 
data = pandas.read_csv("data.csv")
data = data.dropna()

#labels
dataToUse = ['SCL', 'SCRamp' , 'SCRfreq' , 'HRmean' , 'ACCmean', 'Energy' , 'ZCR', 'VoiceProb']
yvals =['3', '4', '5']

xData = data[dataToUse]
yData = data['Hirability']

hireVals = yData.unique()

### 3 FOLD CREATION RANDOMLY 
arr = list( range(len(yData) ))
random.shuffle( arr )
size = (len(arr))


fold1= arr[0:size//3]
fold2= arr[size//3: size*2//3]
fold3 = arr[size*2//3:]

print("THE FOLDS:")
print(fold1)
print(fold2)
print(fold3)
print()

testdata = [fold1, fold2, fold3]

##CROSSS VALIDATION

yPredictedOverall = []
yActualOverall = []

for tests in testdata:
    
    treeFold = RandomForestClassifier(criterion = "entropy", n_estimators=2, max_depth = 3)
    
    xtrain = []
    ytrain = []
    xtest = []
    ytest = []

    for i in range(size):
        if( i not in tests):
            xtrain.append(xData.iloc[i])
            ytrain.append(yData.iloc[i])
        else:
            xtest.append(xData.iloc[i])
            ytest.append(yData.iloc[i])
            

    #print(xtrain, ytrain)

    treeFold = treeFold.fit(xtrain, ytrain)
    plt.figure()
    tree.plot_tree(treeFold.estimators_[0], feature_names = dataToUse, class_names = yvals,rounded = True,filled = True,fontsize=14)
    plt.show()
    tree.plot_tree(treeFold.estimators_[1], feature_names = dataToUse, class_names = yvals,rounded = True,filled = True,fontsize=14)
    plt.show()
   
    print("forest done")
    y_pred = treeFold.predict(xtest)
    print("Actual:", ytest)
    print("Predicted:",y_pred)
    print("Accuracy of the model on the TESTING DATA:",metrics.accuracy_score(ytest, y_pred) ) 

    for i in y_pred:
        yPredictedOverall.append(i)
    for i in ytest:
        yActualOverall.append(i)

    print("Average Absolute Error: ", mean_absolute_error(ytest, y_pred))

    y_pred = treeFold.predict(xtrain)
    print("Accuracy of the model on the TRAINING DATA:",metrics.accuracy_score(ytrain, y_pred) ) 

    
    
print("TOTAL Average Absolute Error: ", mean_absolute_error(yActualOverall, yPredictedOverall))

