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
#Object idea= List of nodes
#Node has (ip,hop,min_rtt,max_rtt,pkts,responses)
#pkt is a dataframe with packets
#hop is from 64-ttl




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

'''
class packet(object):
    rtt=np.NaN
    pkt=np.NaN
    ttl=np.NaN
    def __init__(self,rtt,pkt,ttl):
        self.rtt=rtt
        self.pkt=pkt
        self.ttl=ttl


    def make_packet(rtt,pkt,ttl):
        package=package(rtt,pkt,ttl)
'''

def coojaJsonImporter(dir):

        dataList=[]

        for file in os.listdir(dir):
            print("Importing "+ file)
            with open(dir+"/" + file, 'r') as f:

                dataList.append(json.load(f))

        return dataList

def dict2df(dict):
    dfList=[]
    bigdata=pd.DataFrame()
    for key in dict.keys():
        df=pd.DataFrame(dict[key]['pkts'])
        df['ip']=key

        #dfList.append(df)
        #print(df)
        bigdata=bigdata.append(df,ignore_index=True)

    return bigdata

def dict2df_list(dict):
    dfList=[]
    #dfList(pd.DataFrame(dict))
    for key in dict.keys():
        df=pd.DataFrame(dict[key]['pkts'])
        df['ip']=key
        df['hop']=64-(df['ttl'])
        df = df.drop(['ttl'], axis=1)
        dfList.append(df)

        #print(df)
    return dfList


###Function to create nodes, create a list of nodes
###

def createNodes(dict):
    nodeList=[]
    #dfList(pd.DataFrame(dict))
    for ip in dict.keys():
        pkts=pd.DataFrame(dict[ip]['pkts'])
        #(ip,hop,min_rtt,max_rtt,pkts,responses)
        #print(dict.get(ip).get("max_rtt"))
        #findMissingPackets(dict.get(ip))
        #pkts1=dict.get(ip).get("pkts")
        #pktsList=[]
        #for p in pkts1:
            #print(p.get("rtt"))
             #make_packet(rtt,pkt,ttl)
            #rtt=p.get("rtt")
            #pkt=p.get("pkt")
            #pack=packet(rtt,pkt,ttl)
            #pktsList.append(pack)
        hop=64-(int(pkts[0:1]["ttl"]))
        pkts = pkts.drop(['ttl'], axis=1)
        pkts=pkts.rename(columns={"pkt":"seq"})
        #print(type(pkts[0:1]["ttl"]))
        #print(pkts[0:1]["ttl"])
        n=node(ip,hop,pkts)

        nodeList.append(n)
        #print(type(nodeList[0].pkts[0]))

    return nodeList



def findMissingPackets(node):
    #print(node.pkts["pkt"])
    print("Executed")
    maxP=-1

    for el in node.pkts["pkt"]:
        if(el>maxP): maxP=int(el)
    #print(maxP)
    pkt=[None]*(maxP+1)
    for i in range(len(node.pkts["pkt"])):
        index=int(node.pkts["pkt"][i])
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
    std = df1.std()
    mean = df1.mean()
    a1 = df["rtt"]>mean+(2*std)
    a2 = df["rtt"]<mean-(2*std)
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
    df_a = pd.DataFrame( columns = ['pkt'])
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


