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
from trace_analysis import *
from plots import *



def run(data,directory,colors,cases):
    hops = hopPreparation(data)

    #Distribution of the delay in correlation with the Cases
    #dataHop=hopPreparation(data)
    #Distribution of the delay in correlation with the Hops
    printDensityByCase(directory,data,hops,(15,20),"densitybyCase",colors,cases)

    #Distribution by Hop
    #printDensityByHop(directory,data,(15,20),"densitybyHop",colors,cases)

    #Prints on a file the big matrix (asked by professor)
    printBigPlot(directory,data,(90,20),"Big Plot",colors,cases)

    #Print Density of delay without outliers in every node by Case
    densityOfDelayByCaseNoOutliers(directory,data,(15,90),"Density of delay by Case no outliers",colors,cases)

    #Density of outliers in every node by Case
    densityOutliersByCase(directory,data,(90,20),"Density Outliers of Delay by Case",colors,cases)

    #Distibution of the delay divided by Node in the differents Cases
    densityOfDelayByCase(directory,data,(15,90),"Density of Delay by Case",colors,cases)

    #RTT Graph
    RTTGraph(directory,data,(18,150),"RTT Graph",colors,cases)






colors = [ 'orange','dodgerblue', 'green','violet',"red","yellow","pink"]

directory=os.getcwd() +"/cooja-9nodes/"
cases=[
      "Black Hole Network 1",
        "Black Hole Network 2",
    "Normal Network"
      ]


#data=importCooja(directory)
#run(data,directory,colors,cases)


directory=os.getcwd() + "/cooja-16nodes/"
cases=[
      "Black Hole Network 1",
        "Black Hole Network 2",
       "Black Hole Network 3",
    "Normal Network"
      ]
#data=importCooja(directory)
#run(data,directory,colors,cases)

directory=os.getcwd() + "/iot-lab-25nodes/traces/"
tracefiles=[
    "2019-01JAN-30-1",
    "2019-01JAN-30-1b169",
    "2019-01JAN-30-1b169b153b182",
    "2019-01JAN-30-2",
    "2019-01JAN-30-3b113b122b145b166b185"
]

cases=[
      "2019-01JAN-30-1",
    "2019-01JAN-30-1b169",
    "2019-01JAN-30-1b169b153b182",
    "2019-01JAN-30-2",
    "2019-01JAN-30-3b113b122b145b166b185"
      ]

data = importIOTData(directory,tracefiles)
directory=os.getcwd() + "/iot-lab-25nodes/"
run(data,directory,colors,cases)
