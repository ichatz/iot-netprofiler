import pandas as pd
import matplotlib.pyplot as plt
import os
from node import *


#Import for Kmeans

def import_Cooja2(plots):
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
        "aaaa::212:740a:a:a0a": 4}
    for row in plots:

        #print("Importing ./"+row[0]+"/"+row[1])
        nodeList=import_nodes_Cooja_2(row[0],row[1],node_defaults)
        data.append(nodeList)

    return data
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




#End functions from cooja2 -> nodes for kmeans
