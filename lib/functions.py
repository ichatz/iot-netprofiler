#Modules to install via pip pandas,ipynb
import os
import sys
sys.path.append('../')

import os
import pandas as pd
import numpy as np
import json
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
from pandas.plotting import scatter_matrix
from trace_analysis import *
from node import *
import sklearn.metrics as sm
import pandas as pd
import matplotlib.pyplot as plt
import os
from node import *


from plots_analysis import *
from sklearn.metrics import confusion_matrix
from sklearn.cluster import KMeans
import sklearn.metrics as sm
from sklearn.decomposition import PCA
import random


#Modules to install via pip pandas,ipynb
import sys

sys.path.append('../')

from plots_analysis import *
from sklearn.cluster import KMeans
import pandas as pd
# scipy
import sklearn.metrics as sm


class node(object):
    ip = ""
    hop= 0
    pkts=pd.DataFrame()


    # The class "constructor" - It's actually an initializer
    def __init__(self,ip,hop,pkts):
        self.ip = ip
        self.hop=hop
        self.pkts=pkts

    def make_node(ip,hop,pkts):
        node= node(ip,hop,pkts)
        return node

#Not used anymore
def coojaJsonImporter(dir):

        dataList=[]

        for file in os.listdir(dir):
            print("Importing "+ file)
            with open(dir+"/" + file, 'r') as f:

                dataList.append(json.load(f))

        return dataList


###Function to create nodes, create a list of nodes
###

def createNodes(dict):
    nodeList=[]
    #dfList(pd.DataFrame(dict))
    for ip in dict.keys():
        pkts=pd.DataFrame(dict[ip]['pkts'])

        hop=64-(int(pkts[0:1]["ttl"]))
        pkts = pkts.drop(['ttl'], axis=1)
        pkts=pkts.rename(columns={"pkt":"seq"})
        #print(type(pkts[0:1]["ttl"]))
        #print(pkts[0:1]["ttl"])
        n=node(ip,hop,pkts)

        nodeList.append(n)

    return nodeList



def findMissingPackets(node):
    #print(node.pkts["pkt"])
    print("Executed")
    maxP=-1

    for el in node.pkts["seq"]:
        if(el>maxP): maxP=int(el)
    #print(maxP)
    pkt=[None]*(maxP+1)
    for i in range(len(node.pkts["seq"])):
        index=int(node.pkts["seq"][i])
        #print(index)
        pkt[index]=node.pkts["rtt"][i]
        #pkt[)]=node.pkts["pkt"][i]
    return pkt





def getIps(list):
    ips=[]
    for n in list:
        ips.append(n.ip)
    return ips


def MLPreparation(data):
    # Calculate all the statistics
    statistics = {}	# <node_id, statistics of the node>

    for network in data:
        for node in network:
            print(node.pkts["rtt"].describe())


def getOutliers(df):
    df1=df["rtt"]
    std=df1.std()
    mean=df1.mean()
    a1=df["rtt"]>mean+(2*std)
    a2=df["rtt"]<mean-(2*std)
    return(df[a1 | a2])

def get_IQR_Outliers(df):
    df1 = df["rtt"]
    lower = df1.quantile(.25)
    upper = df1.quantile(.75)
    a1 = df["rtt"]>upper
    a2 = df["rtt"]<lower
    return(df[a1 | a2])

def getStdValues(df):
    df1=df["rtt"]
    std=df1.std()
    mean=df1.mean()
    a1=df["rtt"]<mean+(2*std)
    a2=df["rtt"]>mean-(2*std)
    return(df[a1 & a2])

def getPings(data):
    pings=[]
    for i in range(len(data)):
        packetN=-1
        for j in range(len(data[i])):
            if(len(data[i][j].pkts)>packetN): packetN=len(data[i][j].pkts)
        pings.append(packetN)
    return pings



#Prepare the hop data
def hopPreparation(data):
    hoplist=[]
    df_a = pd.DataFrame( )
    dataHop=[]

    listoflists = []
    #print("Hop Preparation")
    #print(len(data),len(data[0]))

    maxHopCase=[]
    for i in range(len(data)):
        maxHop=-1
        for j in range(len(data[i])):
            if(data[i][j].hop>maxHop):
                maxHop=data[i][j].hop
        maxHopCase.append(maxHop)
    #print(maxHopCase)

    for i in range(len(data)):
        sublist = []
        for j in range(maxHopCase[i]):
            sublist.append((df_a))
        dataHop.append(sublist)
    #print (listoflists)

    for i in range(len(data)):
        col=[]
        for j in range(len(data[i])):
            hop=data[i][j].hop-1

            dataHop[i][hop]= pd.concat([dataHop[i][hop],data[i][j].pkts],sort=True)
    #print(len(dataHop),len(dataHop[0]))

    return dataHop


def getPercentageMissingPackets(node,lenght):
    missing=0
    #print(len(node.pkts))
    missing=lenght-len(node)
    #print(lenght,missing)
    if(missing!=0):
        result=missing/lenght
    else: result=0
    #print(maxS/missing)
    return result*100


def accuracy_score_corrected(correction,labels):
    #print(np.array(correction))
    labels_alt=[]
    sum_labels=0
    sum_labels_alt=0
    for el in labels:
        if (el==0):
            labels_alt.append(1)
            sum_labels_alt+=1
        elif el==1:
            labels_alt.append(0)
            sum_labels+=1

    accuracy=sm.accuracy_score(correction, labels)
    accuracy_alt=sm.accuracy_score(correction, labels_alt)
    #print(correction)


    if (sum_labels>sum_labels_alt):
        #print(accuracy)
        None

    else:
        #print(accuracy_alt)
        labels=labels_alt
    #print(np.array(labels))
    confusionMatrix=sm.confusion_matrix(correction, labels)

    #pprint(confusionMatrix)
    return labels



def ReplaceMissingPackets(node):
    #print(node.pkts["pkt"])
    print("Executed")
    maxP=-1

    for el in node.pkts["seq"]:
        if(el>maxP): maxP=int(el)
    #print(maxP)
    pkt=[None]*(maxP+1)
    for i in range(len(node.pkts["seq"])):
        index=int(node.pkts["seq"][i])
        #print(index)
        pkt[index]=node.pkts["rtt"][i]
        #pkt[)]=node.pkts["pkt"][i]
    return pkt


def import_nodes_Cooja_2(directory,tracemask,node_defaults):
    #print(directory)
    #print(tracemask)
    files = []

    # load all files and extract IPs of nodes
    for file in os.listdir(directory):
        try:
            if file.startswith(tracemask) and file.index("routes"):
                continue
        except:
            files.append(file)

    nodes = pd.DataFrame(columns=['node_id', 'rank'])
    packets_node = {}

    # Load the ICMP traces
    for file in files:
        packets = pd.read_csv(directory + '/' + file,
                              sep=' |icmp_seq=|ttl=|time=',
                              na_filter=True,
                              header=None,
                              skiprows=1,
                              skipfooter=4,
                              usecols=[3, 5, 7, 9],
                              names=['node_id', 'seq', 'hop', 'rtt'],
                              engine='python').dropna().drop_duplicates()

        if len(packets) < 1:
            # Nodes affected by a black hole did not receive any packet
            node_id = file[-24:-4]
            packets = pd.DataFrame(columns=['node_id', 'seq', 'hop', 'rtt'],
                                   data=[[node_id, 1, node_defaults[node_id], 1]])

            nodes.loc[len(nodes)] = [file[-24:-4], node_defaults[node_id]]
            packets_node[file[-24:-4]] = packets

        else:
            #print("qui")
            packets['node_id'] = packets.apply(lambda row: row['node_id'][:-1], axis=1)
            #print(packets["hop"].head())
            #print(nodes)
            #nodes.loc[len(nodes)-1] = [packets['node_id'][0], 64-packets['hop'][0]]
            #print("ciao"+ str(64-packets['hop'][0]))
            #print(nodes.loc[7])
            packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
            packets = packets[packets['rtt'] > 1]
            packets["hop"]=  64-packets['hop']
            packets_node[packets['node_id'][0]] = packets

    nodes=nodes.sort_values(by=['rank', 'node_id'])

    #tranformation in node
    nodeList=[]

    for n in packets_node.keys():
        #print((packets_node[n]).head())
        pkts=packets_node[n].drop(["node_id","hop"],axis=1)
        #print(pkts)
        hop=int(packets_node[n]["hop"][0])
        ip=packets_node[n]["node_id"][0]
        #print(hop)
        n=node(ip,hop,pkts)
        nodeList.append(n)


    return nodeList


def import_Cooja2(df,directory):
    data=[]
    node_defaults = {
        "aaaa::212:7403:3:303": 1,
        "aaaa::212:7402:2:202": 2,
        "aaaa::212:7404:4:404": 2,
        "aaaa::212:7406:6:606": 2,
        "aaaa::212:7405:5:505": 3,
        "aaaa::212:7407:7:707": 3,
        "aaaa::212:7409:9:909": 3,
        "aaaa::212:7408:8:808": 4,
        "aaaa::212:740a:a:a0a": 4,
        "aaaa::212:740b:b:b0b": 10,
    "aaaa::212:740f:f:f0f":10,
        "aa::212:7411:11:1111":10,
"aaaa::212:740d:d:d0d":10,




    }
    #for row in plots:

        #print("Importing ./"+row[0]+"/"+row[1])
    #print(directory+df["directory"].values)

    for i in range(len(df["directory"].values)):




        nodeList=import_nodes_Cooja_2(directory+df["directory"].values[i],df["case"].values[i],node_defaults)
        data.append(nodeList)
    #print(len(data))
    #print(len(data[0]))
    return data


def analyze_network(directory, df, pings, window, features_to_drop):
    cases = []
    casesAccuracy = df["case_accuracy"].values
    casesAccuracy2 = df["case_accuracy2"].values
    #     for row in plots:
    #         cases.append(row[1])
    #         casesAccuracy.append(row[2])
    #         data=import_Cooja2(plots)
    cases = df["case"].values
    folder = df["directory"].values + directory

    data = import_Cooja2(df, directory)

    # pings=getPings(data)
    # All data collection is in variable node that is a list of list of nodes
    # 3 nets input x 9 nodes by net
    print("Processing...")
    d = {"label": [],
         "type": [],
         "count": [],
         "std": [],
         "mean": [],
         "var": [],
         "hop": [],

         "packet loss": [],
         "outliers": [],
         "node": [],
         "window": []
         }
    # count=[]
    labels = []
    var = []
    # window=100
    # stats=pd.DataFrame(columns=columns)
    n = pings

    for i in range(len(data)):
        # window=pings[i]

        for j in range(len(data[i])):
            # n=pings[i]

            # print(n)
            for z in range(0, n, int(window)):
                # if(z+window>n):break
                # print(z,z+window)

                # df1 = df1.assign(e=p.Series(np.random.randn(sLength)).values)
                node = data[i][j].pkts
                name = str(j) + " " + cases[i]
                nodeWindow = node[(node["seq"] < z + window) & (node["seq"] >= z)]
                nodeWindowP = nodeWindow["rtt"]
                d["count"].append(nodeWindowP.count())
                # Case without outliers
                # Case with outliers
                std = 0
                if (nodeWindowP.std() > 10):
                    std = 1
                    std = nodeWindowP.std()

                d["std"].append(std)
                mean = nodeWindowP.mean()
                # if(mean<1):print(mean)
                d["mean"].append(mean)
                var = 0
                if (nodeWindowP.var() > var): var = nodeWindowP.var()
                d["var"].append(var)
                d["label"].append(cases[i])
                d["hop"].append(data[i][j].hop)
                d["type"].append(casesAccuracy[i])
                d["outliers"].append(getOutliers(nodeWindow)["rtt"].count())
                missing = window - nodeWindow.count()
                d["node"].append(data[i][j].ip)
                mP = getPercentageMissingPackets(nodeWindow, window)
                PL = 0
                if (mP > 30):
                    PL = 1
                    PL = mP
                d["packet loss"].append(mP)
                d["window"].append(window)

    stats = pd.DataFrame(d)

    dataK = stats.drop(features_to_drop, axis=1)
    dataK = dataK.fillna(0)

    # print(dataK)
    correction = []
    correction_alt = []
    col = np.array(dataK["type"])
    dataK = dataK.drop(["type"], axis=1)
    # Creating simple array to correct unsupervised learning
    # NB as it is unsupervised could happen that the correction are inverted
    for i in range(len(col)):
        el = d["type"][i]
        if el == "normal":
            correction.append(1)
            correction_alt.append(0)

        else:

            correction.append(0)
            correction_alt.append(1)

    dataC = stats["label"]
    kmeans = KMeans(n_clusters=2)
    kmeans.fit(dataK)
    labels = kmeans.predict(dataK)
    centroids = kmeans.cluster_centers_
    labels = accuracy_score_corrected(correction, labels)
    predicted = []
    for i in range(len(labels)):

        if (labels[i] == 1):
            predicted.append("normal")
        else:
            predicted.append("BH")

    # print(len(predicted))
    stats["predicted"] = pd.Series(np.array(predicted))
    stats["predicted number"] = pd.Series(np.array(labels))
    stats["correction number"] = pd.Series(np.array(correction))
    stats_csv = stats[[
        "label",
        "type",
        "predicted",
        "packet loss",
        "outliers",
        "std",
        "hop",
        "node",
        "mean"

    ]]
    # stats_csv.to_csv("results_kmeans.csv", sep='\t', encoding='utf-8')
    stats.head()
    net_results = {
        "case": [],
        "normal_behaving_nodes_percentage": [],
        "predicted": [],
        "real": [],
        "pings": [],
        "window": [],

    }
    # print(stats["predicted number"])
    correction = []
    labels = []
    for case in range(len(cases)):
        subset = stats[stats["label"] == cases[case]]
        mean_predicted = str(subset["predicted number"].mean() * 100)  # +"% normal"
        net_results["case"].append(cases[case])
        net_results["normal_behaving_nodes_percentage"].append(mean_predicted)
        net_results["pings"].append(pings)
        net_results["window"].append(window)

        if (float(mean_predicted) < 85):
            p = "abnormal"
            labels.append(0)
        else:
            p = "normal"
            labels.append(1)

        if (casesAccuracy[case] == "BH"):
            c = "abnormal"
            correction.append(0)
        elif (casesAccuracy[case] == "normal"):
            c = "normal"
            correction.append(1)

        net_results["predicted"].append(p)
        net_results["real"].append(c)

    results = pd.DataFrame(net_results)

    return results, stats, correction, labels


def get_traces_csv(directory):
    print("Reading Traces from " + directory)
    directory1 = directory
    directory += "traces/"
    # print(directory)
    files = []
    path = []
    case_accuracy = []
    case_accuracy2 = []
    # directory="./traces"
    # directory=os.getcwd()+"/traces/"
    d = {}
    try:

        for subdirectory in os.listdir(directory):
            # print(os.path.isdir(subdirectory))
            # print(subdirectory)

            subdirectory2 = directory + "/" + subdirectory

            if (os.path.isdir(subdirectory2)):

                for file in os.listdir(subdirectory2):

                    if ("routes" in file):
                        # print(subdirectory+"/"+file)
                        path.append("traces/" + subdirectory[:])
                        # print(file)
                        files.append(file[:-10])
                        # print(file)
                        if ("normal" in file):
                            case_accuracy.append("normal")
                            case_accuracy2.append("normal")
                        elif ("bh" in file):
                            case_accuracy.append("BH")
                            case_accuracy2.append("BH")
                        elif ("gh" in file):
                            case_accuracy.append("BH")
                            case_accuracy2.append("GH")
                        continue
        d = {
            "directory": path,
            "case": files,
            "case_accuracy": case_accuracy,
            "case_accuracy2": case_accuracy2

        }
    except:
        pass

    traces = pd.DataFrame(d)
    # print(directory)
    traces.to_csv(directory + "traces.csv", sep=',', encoding='utf-8')
    return traces


def run(directory, df):
    colors = [
        'orange', 'dodgerblue',
        'forestgreen', 'violet',
        "red", "brown",
        "pink", "aqua",
        "darkslategrey", "darkred",
        "darkblue", "darkorchid",
        "salmon", "chocolate"

    ]
    casesAccuracy = df["case_accuracy"].values

    cases = df["case"].values
    folder = df["directory"].values + directory

    data = import_Cooja2(df, directory)

    # hops = hopPreparation(data)

    # Distribution of the delay in correlation with the Cases
    # dataHop=hopPreparation(data)
    # Distribution of the delay in correlation with the Hops
    # printDensityByCase(directory,data,hops,(15,90),"densitybyCase",colors,cases)

    # Distribution by Hop
    # printDensityByHop(directory,data,hops,(30,90),"densitybyHop",colors,cases)

    # Prints on a file the big matrix (asked by professor)
    # printBigPlot(directory,data,(90,90),"Big Plot",colors,cases)

    # Print Density of delay without outliers in every node by Case
    # densityOfDelayByCaseNoOutliers(directory,data,(15,90),"Density of delay by Case no outliers",colors,cases)

    # Density of outliers in every node by Case
    # densityOutliersByCase(directory,data,(90,90),"Density Outliers of Delay by Case",colors,cases)

    # Distibution of the delay divided by Node in the differents Cases
    # densityOfDelayByCase(directory,data,(15,90),"Density of Delay by Case",colors,cases)

    # RTT Graph
    RTTGraph(directory, data, (30, 90), "RTT Graph", colors, cases)


def create_results(directory, features_to_drop):
    df = pd.read_csv(directory + "/traces/traces.csv", sep=',', encoding='utf-8')
    accuracy = {
        "Model": [],
        "Window Size": [],
        "Accuracy": []

    }
    window_size = [25, 50, 100]
    results_total = pd.DataFrame()
    results_total_nodes = pd.DataFrame()
    for i in window_size:
        results_kmeans_network, results_kmeans_node, correction, labels = analyze_network(directory, df, 100, i,
                                                                                          features_to_drop)
        results_total = results_total.append(results_kmeans_network, ignore_index=True)
        results_total_nodes = results_total_nodes.append(results_kmeans_node, ignore_index=True)

        # Accuracy per node
        # labels = results_kmeans_node["predicted number"].values
        # correction = results_kmeans_node["correction number"].values
        # accuracy["Accuracy"].append(sm.accuracy_score(correction, labels))

        # Accuracy per network
        accuracy["Accuracy"].append(sm.accuracy_score(correction, labels))

        accuracy["Model"].append("Kmeans")
        accuracy["Window Size"].append(i)

    accuracy = pd.DataFrame(accuracy)

    results_total.to_csv(directory + "results_total.csv", sep=',', encoding='utf-8')
    accuracy.to_csv(directory + "accuracy.csv", sep=',', encoding='utf-8')
    results_total_nodes.to_csv(directory + "results_total_node.csv", sep=',', encoding='utf-8')

    return results_total, accuracy, results_total_nodes