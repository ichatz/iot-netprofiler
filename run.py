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


def run(directory,df):
    colors = [
    'orange','dodgerblue',
     'forestgreen','violet',
     "red","brown",
     "pink","aqua",
     "darkslategrey","darkred",
     "darkblue","darkorchid",
     "salmon","chocolate"

     ]
    casesAccuracy=df["case_accuracy"].values

    cases=df["case"].values
    folder=df["directory"].values+directory

    data=import_Cooja2(df,directory)

    #hops = hopPreparation(data)

    #Distribution of the delay in correlation with the Cases
    #dataHop=hopPreparation(data)
    #Distribution of the delay in correlation with the Hops
    #printDensityByCase(directory,data,hops,(15,90),"densitybyCase",colors,cases)

    #Distribution by Hop
    #printDensityByHop(directory,data,hops,(30,90),"densitybyHop",colors,cases)

    #Prints on a file the big matrix (asked by professor)
    #printBigPlot(directory,data,(90,90),"Big Plot",colors,cases)

    #Print Density of delay without outliers in every node by Case
    #densityOfDelayByCaseNoOutliers(directory,data,(15,90),"Density of delay by Case no outliers",colors,cases)

    #Density of outliers in every node by Case
    #densityOutliersByCase(directory,data,(90,90),"Density Outliers of Delay by Case",colors,cases)

    #Distibution of the delay divided by Node in the differents Cases
    #densityOfDelayByCase(directory,data,(15,90),"Density of Delay by Case",colors,cases)

    #RTT Graph
    RTTGraph(directory,data,(30,90),"RTT Graph",colors,cases)





#https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib


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

print(directory)




directory=os.getcwd()+"/cooja3-9nodes/"
#directory="../cooja3-9nodes/"

df=pd.read_csv(directory+"/traces/traces.csv", sep=',', encoding='utf-8')
#print(directory+df["directory"].values)
#print(directory)
#analyze_network(directory,df,200,50)
run(directory,df)
#analyze_network(directory,plots,200,25)
