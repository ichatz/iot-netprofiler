#Modules to install via pip pandas,ipynb


#Modules to install via pip pandas,ipynb
import sys

sys.path.append('../')

from plots_analysis import *
# scipy

import random
random.seed(6666)

sys.path.append('../lib/')

from plots_analysis import *
from lib.functions import *

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
df=pd.read_csv(directory+"/traces/traces.csv", sep='\t', encoding='utf-8')
#df=get_traces_csv(directory)


#print(df)

results_total=pd.DataFrame()


results_kmeans_network,results_kmeans_node=analyze_network(directory,df,200,25)
results_total=results_total.append(results_kmeans_network,ignore_index = True)
results_kmeans_network,results_kmeans_node=analyze_network(directory,df,200,50)
results_total=results_total.append(results_kmeans_network,ignore_index = True)
results_kmeans_network,results_kmeans_node=analyze_network(directory,df,200,100)
results_total=results_total.append(results_kmeans_network,ignore_index = True)
results_kmeans_network,results_kmeans_node=analyze_network(directory,df,200,200)
results_total=results_total.append(results_kmeans_network,ignore_index = True)


results_total.sort_values('case')
print(results_total)

results_total.to_csv(directory+"results_total.csv", sep='\t', encoding='utf-8')
#results_kmeans.to_csv(directory+"results_network_kmeans.csv", sep=',', encoding='utf-8')





