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

# arr = list( range(len(stamina) ))
# random.shuffle( arr )

# size = len(arr)//5

# #random values 
# test1 = arr[0:size]
# test2 = arr[size: size*2]
# test3 = arr[size*2: size*3]
# test4 = arr[size*3: size*4]
# test5 = arr[size*4:: ] #has the extra value 

# testdata = [test1, test2, test3, test4, test5]
# print(testdata)
# print(test1)
# print(test2)
# print(test3)
# print(test4)
# print(test5)

#keeping
test1 = [24, 96, 85, 120, 67, 145, 37, 59, 78, 6, 122, 54, 1, 106, 0, 43, 140, 31, 87, 101, 102, 15, 86, 33, 5, 65, 105, 61, 97]
test2 = [13, 119, 28, 34, 58, 64, 3, 131, 9, 55, 22, 26, 2, 142, 108, 113, 141, 44, 49, 95, 11, 123, 73, 98, 56, 36, 114, 103, 47]
test3 = [23, 77, 134, 128, 48, 40, 53, 115, 124, 116, 126, 41, 45, 27, 69, 121, 50, 42, 99, 71, 100, 111, 129, 14, 88, 8, 80, 83, 39]
test4 = [104, 4, 52, 84, 133, 7, 107, 138, 91, 90, 46, 18, 144, 130, 93, 30, 136, 60, 117, 76, 139, 68, 66, 38, 63, 19, 72, 16, 70]
test5 = [21, 75, 51, 10, 118, 132, 94, 127, 29, 25, 112, 12, 109, 32, 82, 110, 92, 135, 79, 89, 57, 35, 20, 17, 62, 74, 81, 143, 125, 137]
# test1 = [127, 99, 106, 72, 22, 13, 61, 53, 139, 110, 97, 11, 47, 132, 44, 43, 140, 86, 113, 14, 114, 70, 133, 23, 76, 66, 9, 50, 73]
# test2 = [131, 46, 77, 2, 145, 7, 118, 120, 65, 84, 36, 108, 1, 129, 58, 60, 31, 40, 88, 81, 80, 69, 122, 67, 130, 0, 102, 74, 59]
# test3 = [126, 27, 103, 64, 39, 32, 121, 4, 116, 26, 91, 78, 92, 33, 62, 51, 10, 85, 143, 38, 75, 5, 115, 8, 41, 18, 19, 35, 21]
# test4 = [54, 134, 144, 89, 90, 37, 48, 63, 100, 57, 15, 117, 96, 93, 29, 119, 17, 101, 49, 111, 87, 12, 95, 104, 82, 45, 94, 24, 123]
# test5 = [137, 6, 128, 68, 56, 98, 83, 16, 55, 125, 52, 109, 79, 136, 71, 30, 42, 141, 124, 20, 28, 25, 34, 105, 138, 142, 107, 3, 135, 112]
testdata = [test1, test2, test3, test4, test5]
count = 1
RSSAll = []
for tests in testdata: 
    print("This is with cross-validation: ", count)
    X = numpy.empty((0, 5), float) 
    Y = numpy.empty((0,1), float)
    RSS_data = numpy.empty( (0,5), float)
    actual = numpy.empty( (0,1), float)

    for i in range(len(stamina)):
        #first when test1 is test : check if index is one of the ones in test1,2,3,4,5
        if i not in tests:
            #data needs to be training data
            #not including capture rate 
            xsample = numpy.array( [1, stamina[i], attack_value[i], defense_value[i], spawn_channel[i]] ) 
            X = numpy.append(X, [xsample], axis=0)
            Y = numpy.append(Y, [ [combat_point[i]] ], axis=0)
        else:
            rss_sample = numpy.array( [1, stamina[i], attack_value[i], defense_value[i],  spawn_channel[i]] ) 
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
   


        



