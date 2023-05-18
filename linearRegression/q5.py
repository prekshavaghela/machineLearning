import csv
import math
from unittest import TestLoader
import matplotlib.pyplot as plt
import random
import numpy

file = open('hw2_data_edit.csv')
data = csv.reader(file)

#read all the values and get the data. deleted the name column. 
stamina = []
attack_value = []
defense_value = []
capture_rate = []
flee_rate = []
spawn_channel = []
combat_point = []

for vals in data: 
    stamina.append( int(vals[0]) )
    attack_value.append( int(vals[1]) )
    defense_value.append( int(vals[2]) )
    capture_rate.append( float(vals[3]) )
    flee_rate.append( float(vals[4]) )
    spawn_channel.append( float(vals[5]) )
    combat_point.append( int(vals[6]) )

arr = list( range(len(stamina) ))
random.shuffle( arr )
size = len(arr)//5

#random values 
test1 = arr[0:size]
test2 = arr[size: size*2]
test3 = arr[size*2: size*3]
test4 = arr[size*3: size*4]
test5 = arr[size*4:: ] #has the extra value 

testdata = [test1, test2, test3, test4, test5]

print(test1)
print(test2)
print(test3)
print(test4)
print(test5)

#mean
meanCombatPoint = sum(combat_point)/len(combat_point)
print(meanCombatPoint)

acc_array = []
count = 1

for tests in testdata: 
    print("This is with cross-validation: ", count)
    w = numpy.array( [1,1,1,1,1,1,1] )
    w = w.transpose()

    for k in range(150):

        i = random.randint(0, len(stamina)-1 )
        
        #first when test1 is test : check if index is one of the ones in test1,2,3,4,5
        if i not in tests:

            #data needs to be training data
            x_vals = numpy.array( [1, stamina[i], attack_value[i], defense_value[i], capture_rate[i], flee_rate[i], spawn_channel[i]] )
            predY = numpy.dot(w, x_vals )
            
            #check if the value side of the line
            if( predY - meanCombatPoint > 0):
                y_predict = 1
            else:
                y_predict = -1

            if( combat_point[i] - meanCombatPoint > 0):
                y_actual = 1
            else: 
                y_actual = -1
                
            if ( y_predict != y_actual):
                #update the weight
                w = w + y_actual * x_vals
                #shift

    print("weights: " , w)

    correct = 0 

    for i in tests:
        xvalue = numpy.array( [1, stamina[i], attack_value[i], defense_value[i], capture_rate[i], flee_rate[i], spawn_channel[i]] )
        predY = numpy.dot(w, xvalue )
    
        if( predY - meanCombatPoint > 0):
            y_predict = 1
        else:
            y_predict = -1

        if( combat_point[i] - meanCombatPoint > 0):
            y_actual = 1
        else: 
            y_actual = -1
                
        if( y_actual == y_predict):
            correct +=1
        
    
    accuracy = correct/len(tests)
    print("accuracy of test ", count, " : ", accuracy)
    acc_array.append(accuracy)
    count+=1


print( "total accuracy: ", sum(acc_array)/len(acc_array) )
          

        



        



