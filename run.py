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
from trace_analysis_cooja2 import *
from node import *


def run(data,directory,colors,cases):
    hops = hopPreparation(data)

    #Distribution of the delay in correlation with the Cases
    dataHop=hopPreparation(data)
    #Distribution of the delay in correlation with the Hops
    printDensityByCase(directory,data,hops,(15,20),"densitybyCase",colors,cases)

    #Distribution by Hop
    printDensityByHop(directory,data,hops,(15,20),"densitybyHop",colors,cases)

    #Prints on a file the big matrix (asked by professor)
    printBigPlot(directory,data,(90,20),"Big Plot",colors,cases)

    #Print Density of delay without outliers in every node by Case
    densityOfDelayByCaseNoOutliers(directory,data,(15,90),"Density of delay by Case no outliers",colors,cases)

    #Density of outliers in every node by Case
    densityOutliersByCase(directory,data,(90,20),"Density Outliers of Delay by Case",colors,cases)

    #Distibution of the delay divided by Node in the differents Cases
    densityOfDelayByCase(directory,data,(15,90),"Density of Delay by Case",colors,cases)

    #RTT Graph
    RTTGraph(directory,data,(30,150),"RTT Graph",colors,cases)






colors = [ 'orange','dodgerblue', 'green','violet',"red","brown","pink"]

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

#data = importIOTData(directory,tracefiles)
#directory=os.getcwd() + "/iot-lab-25nodes/"
#run(data,directory,colors,cases)
directory=os.getcwd()+"/cooja2-9nodes/"
plots = [(directory+"traces/normal", 'grid9_normal_2019-02-11_17:51:17_'),
         (directory+"traces/normal", 'grid9_normal_2019-02-11_20:22:01_'),
         (directory+"traces/1bh-6", 'grid9_1bh-6_2019-02-11_20:48:08_'),
         (directory+"traces/1bh-6", 'grid9_1bh-6_2019-02-11_21:03:19_'),
         (directory+"traces/1bh-3", 'grid9_1bh-3_2019-02-12_14:47:14_')
        ]

print(directory)
cases=[
    "grid9_normal_2019-02-11_17:51:17_",
    'grid9_normal_2019-02-11_20:22:01_',
    'grid9_1bh-6_2019-02-11_20:48:08_',
    'grid9_1bh-6_2019-02-11_21:03:19_',
    'grid9_1bh-3_2019-02-12_14:47:14_'
]

#data=import_Cooja2(plots)

#run(data,directory,colors,cases)
directory=os.getcwd()+"/cooja3-9nodes/"
plots = [
        (directory+"traces/1bh-3", 'grid9_1bh-3_2019-02-13_16:28_'),
         (directory+"traces/1bh-5", 'grid9_1bh-5_2019-02-13_15:31_'),
        #(directory+"traces/1bh-6", 'grid9_1bh-6_2019-02-13_12:59_'),
         #(directory+"traces/1bh-7", 'grid9_1bh-7_2019-02-13_15:08_'),
         #(directory+"traces/1bh-9", 'grid9_1bh-9_2019-02-13_15:57_'),

        # (directory+"traces/normal", 'grid9_normal_2019-02-13_17:05_')
        ]

print(directory)
cases=[
    'grid9_1bh-3_2019-02-13_16:28_',
    'grid9_1bh-5_2019-02-13_15:31_',
     'grid9_1bh-6_2019-02-13_12:59_',
     'grid9_1bh-7_2019-02-13_15:08_',
      'grid9_1bh-9_2019-02-13_15:57_',
      'grid9_normal_2019-02-13_17:05_'
]
data=import_Cooja2(plots)
# for i in range(len(data)):
#     for j in range(len(data[i])):
#         print(data[i][j].hop)

run(data,directory,colors,cases)
