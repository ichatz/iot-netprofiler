 
#Modules to install via pip pandas,ipynb
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import json
from pprint import pprint
import os
import import_ipynb
import sys
sys.path.append('../')
from functions import *
from pandas.plotting import scatter_matrix



directory="/home/fedebyes/Workspace/Master Thesis/iot-netprofiler/"
directory=directory+"cooja-9nodes/"
traces=directory+"traces"

dataList=coojaJsonImporter(traces)
test1BH1=dataList[0]
test1BH2=dataList[1]
testNorm =dataList[2]
data=[]
cases=[
      "Black Hole Network 1",
        "Black Hole Network 2",
    "Normal Network"
      ]
colors = [ 'orange','dodgerblue', 'green','violet']
for nodeList in dataList:
    data.append(createNodes(nodeList))


#All data collection is in variable node that is a list of list of nodes
#3 nets input x 9 nodes by net
data[0][0].pkts[1:5]

print(getPings(data))

#Distribution of the delay in correlation with the Cases
dataHop=hopPreparation(data)
#Distribution of the delay in correlation with the Hops
printDensityByCase(directory,data,(15,20),"densitybyCase",colors,cases)

#Distribution by Hop
printDensityByHop(directory,data,(15,20),"densitybyHop",colors,cases)

#Prints on a file the big matrix (asked by professor)
printBigPlot(directory,data,(15,20),"Big Plot",colors,cases)

#Print Density of delay without outliers in every node by Case
densityOfDelayByCaseNoOutliers(directory,data,(15,90),"Density of delay by Case no outliers",colors,cases)

#Density of outliers in every node by Case
densityOutliersByCase(directory,data,(30,10),"Density Outliers of Delay by Case",colors,cases)

#Distibution of the delay divided by Node in the differents Cases
densityOfDelayByCase(directory,data,(15,90),"Density of Delay by Case",colors,cases)


#RTT Graph
RTTGraph(directory,data,(18,70),"RTT Graph",colors,cases)






