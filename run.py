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
from plots_analysis import *
from trace_analysis_cooja2 import *
from node import *




#https://stackoverflow.com/questions/22408237/named-colors-in-matplotlib

""""
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
df=get_traces_csv(directory)
#print(directory)
"""



#directory=os.getcwd()+"/cooja3-9nodes/"
#df=get_traces_csv(directory)
#print(df)

#results_kmeans=analyze_network(directory,df,200,50)
#run(df,directory,colors,cases)
#df=pd.read_csv(directory+"/traces/traces.csv", sep=',', encoding='utf-8')
#print(results_kmeans)


directory=os.getcwd()+"/cooja3-9nodes/"
df=get_traces_csv(directory)
results_kmeans=analyze_network(directory,df,200,50)
print(results_kmeans)
