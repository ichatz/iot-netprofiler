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

def saveFileFigures(fig,directory,namefile):
    directory=directory+"figures/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(directory)
    fig.savefig(directory+namefile+".pdf")   # save the figure to file
    #plt.show()


#Prints on a file the big matrix (asked by professor)
def printBigPlot(directory,data,figsize,namefile,colors,cases):
    print("Printing Big Plot for "+directory)
    fig, axs= plt.subplots(len(data),len(data[0]), figsize=figsize,sharey=True, )

    for i in range(len(data)):
        for j in range(len(data[i])):
        #print(i,j)
            ax=axs[i][j]
            data[i][j].pkts["rtt"].plot.kde(
                ax=ax,
                label="Case " +str(cases[i]),
                color=colors[i]

            )

            ax.set_ylabel("Density")
            data[i][j].pkts["rtt"].hist(density=True,alpha=0.3,color=colors[i], ax=ax)
            ax.set_title("Node "+ str(j) )
            ax.set_xlabel("Time (ms)")
            ax.legend()
            ax.set_xlim([-500, 8000])
    saveFileFigures(fig,directory,namefile)

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

#Print on a file density by Hop (asked by professor)
def printDensityByHop(directory,data,figsize,namefile,colors,cases):

    print("Printing Density by Hop for "+directory)
    dataHop=hopPreparation(data)
    fig, axs= plt.subplots(len(dataHop),1, figsize=(15,20),sharey=True, )
    for i in range(len(dataHop)):
        for j in range(len(dataHop[i])):
            dataHop[i][j]['rtt'].plot.kde(
                ax=axs[j],
                label=cases[i],color=colors[i]
            )

            dataHop[i][j]["rtt"].hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])

            axs[j].set_xlabel("Time (ms)")
            axs[j].set_title("Hop "+ str(j+1))
            axs[j].legend()

            axs[j].set_xlim([-40, 6000])
    saveFileFigures(fig,directory,namefile)

#Print on a file density by Case (asked by professor)
def printDensityByCase(directory,data,figsize,namefile,colors,cases):

    print("Printing Density by case for "+directory)
    #print(len(data),len(data[0]))

    data1=hopPreparation(data)
    dataHopT=[*zip(*data1)]

    #print(len(data1),len(data1[0]))
    #print(len(dataHopT),len(dataHopT[0]))
    fig, axs= plt.subplots(len(dataHopT[0]),1, figsize=(15,20),sharey=True, )
    for i in range(len(dataHopT)):
        for j in range(len(dataHopT[0])):

            dataHopT[i][j]["rtt"].plot.kde(
                ax=axs[j],
                label="Hop "+str(i),
                color=colors[i]
            )

            dataHopT[i][j]["rtt"].hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])
            axs[j].set_title(""+ cases[i-j])
            axs[j].set_xlabel("Time (ms)")
            axs[j].legend()

            axs[j].set_xlim([-40, 6000])
    saveFileFigures(fig,directory,namefile)

#Print Density of delay without outliers in every node by Case
def densityOfDelayByCaseNoOutliers(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of delay without outliers in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            out=getStdValues(data[i][j].pkts)
            if not out.empty :
                ax=axs[j]
                out["rtt"].plot.kde(
                ax=ax,
                label=cases[i],
                     color=colors[i]
            )
                ax.set_ylabel("Density")
                out["rtt"].hist(density=True,alpha=0.3, ax=ax, color=colors[i])
                ax.set_title("Node "+ str(j))
                ax.set_xlabel("Time (ms)")
                ax.legend()
    saveFileFigures(fig,directory,namefile)

#Density of outliers in every node by Case
def densityOutliersByCase(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of outliers in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data),len(data[0]), figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            out=getOutliers(data[i][j].pkts)
            if not out.empty :
                ax=axs[i][j]
                out["rtt"].plot.kde(
                ax=ax,
                label=cases[i],
                 color=colors[i]
            )
                ax.set_ylabel("Density")
                out["rtt"].hist(density=True,alpha=0.3, ax=ax, color=colors[i])
                ax.legend()
            ax.set_title("Node "+ str(j))
            ax.set_xlabel("Time (ms)")

    saveFileFigures(fig,directory,namefile)


#Distibution of the delay divided by Node in the differents Cases
def densityOfDelayByCase(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of delay in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            data[i][j].pkts["rtt"].plot.kde(
                ax=axs[j],
                label=cases[i],color=colors[i]
            )
            axs[j].set_ylabel("Time (ms)")
            data[i][j].pkts["rtt"].hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])
            axs[j].set_title("Node "+ str(j))
            axs[j].set_xlabel("Time (ms)")
            axs[j].legend()
    saveFileFigures(fig,directory,namefile)


#RTT Graph
def RTTGraph(directory,data,figsize,namefile,colors,cases):
    print("Printing RTT Graph for "+directory)
    fig, axs= plt.subplots(9,1, figsize=(18,70),sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            axs[j].plot(data[i][j].pkts["pkt"],data[i][j].pkts["rtt"],label=cases[i],color=colors[i]   )
            axs[j].set_title("Node "+ str(j))
            axs[j].set_xlabel("Packet Number")
            axs[j].set_ylabel("Time (ms)")
            axs[j].legend()
    saveFileFigures(fig,directory,namefile)
