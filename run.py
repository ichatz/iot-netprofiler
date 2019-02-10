 
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



def runCooja(directory,colors,cases):
    data=[]
    traces=directory+"traces"
    dataList=coojaJsonImporter(traces)
    for nodeList in dataList:
        data.append(createNodes(nodeList))

    
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


colors = [ 'orange','dodgerblue', 'green','violet']

directory="/home/fedebyes/Workspace/Master Thesis/iot-netprofiler/"
directory=directory+"cooja-9nodes/"
cases=[
      "Black Hole Network 1",
        "Black Hole Network 2",
    "Normal Network"
      ]
#print(getPings(data))

#runCooja(directory,colors,cases)

directory="/home/fedebyes/Workspace/Master Thesis/iot-netprofiler/"
directory=directory+"cooja-16nodes/"
cases=[
      "Black Hole Network 1",
        "Black Hole Network 2",
       "Black Hole Network 3",
    "Normal Network"
      ]
runCooja(directory,colors,cases)
