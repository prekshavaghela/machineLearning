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
print(testdata)

print(test1)
print(test2)
print(test3)
print(test4)
print(test5)

count = 1
RSSAll = []
for tests in testdata: 
    print("This is with cross-validation: ", count)
    X = numpy.empty((0, 7), float) 
    Y = numpy.empty((0,1), float)
    RSS_data = numpy.empty( (0,7), float)
    actual = numpy.empty( (0,1), float)

    for i in range(len(stamina)):
        #first when test1 is test : check if index is one of the ones in test1,2,3,4,5
        if i not in tests:
            #data needs to be training data
            xsample = numpy.array( [1, stamina[i], attack_value[i], defense_value[i], capture_rate[i], flee_rate[i], spawn_channel[i]] ) 
            X = numpy.append(X, [xsample], axis=0)
            Y = numpy.append(Y, [ [combat_point[i]] ], axis=0)
        else:
            rss_sample = numpy.array( [1, stamina[i], attack_value[i], defense_value[i], capture_rate[i], flee_rate[i], spawn_channel[i]] ) 
            RSS_data = numpy.append(RSS_data, [rss_sample], axis=0)
            actual = numpy.append( actual, [ [combat_point[i]]], axis=0)
    #ORDINARY LEAST SQUARES SOLUTION:
    # W* = (X^T*X)^-1 * X^T*y
    w = numpy.dot( numpy.dot( numpy.linalg.inv( numpy.dot( X.transpose(), X) ), X.transpose() ) , Y)
    print( "weights: ", len(w))
    print(w)

    
    #use the data from test: RSS_data and actual 
    one_part = actual - numpy.dot( RSS_data, w)
    RSS_error = numpy.dot( one_part.transpose(), one_part)
    print( "square root of RSS-error :", math.sqrt(RSS_error) )
    print()
    RSSAll.append(float(math.sqrt( RSS_error[0]) ) )

    count+=1


print("Average RSS ERROR: ", sum(RSSAll)/len(RSSAll) )
   


        



