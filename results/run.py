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



#directory=os.getcwd()+"/cooja3-9nodes/"
#df=get_traces_csv(directory)
#print(df)

#results_kmeans=analyze_network(directory,df,200,50)
#run(df,directory,colors,cases)
#df=pd.read_csv(directory+"/traces/traces.csv", sep=',', encoding='utf-8')
#print(results_kmeans)


directory=os.getcwd()+"/cooja3-16nodes/"
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

results_total.to_csv(directory+"results_total.csv", sep='\t', encoding='utf-8',index_col = None)
#results_kmeans.to_csv(directory+"results_network_kmeans.csv", sep=',', encoding='utf-8')

##################################
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

results_total.to_csv(directory+"results_total.csv", sep='\t', encoding='utf-8',index_col = None)


