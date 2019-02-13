import pandas as pd
import matplotlib.pyplot as plt
import os
from node import *


#Import for Kmeans

def import_Cooja2(plots):
    data=[]

    for row in plots:

        #print("Importing ./"+row[0]+"/"+row[1])
        nodeList=import_nodes_Cooja_2(row[0],row[1])
        data.append(nodeList)

    return data
def import_nodes_Cooja_2(directory,tracemask):
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


        nodes.loc[len(nodes)] = [packets['node_id'][0], 64 - packets['hop'][0]]


        packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
        packets = packets[packets['rtt'] > 1]




        #print(data)
        packets_node[packets['node_id'][0]] = packets
        nodeList=[]

        for n in packets_node.keys():
            #print((packets_node[n]).head())
            pkts=packets_node[n].drop(["node_id","hop"],axis=1)
            hop=64-int(packets_node[n]["hop"][0])
            ip=packets_node[n]["node_id"][0]
            n=node(ip,hop,pkts)
            nodeList.append(n)


    return nodeList




#End functions from cooja2 -> nodes for kmeans
