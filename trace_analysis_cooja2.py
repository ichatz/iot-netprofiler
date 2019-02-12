import pandas as pd
import matplotlib.pyplot as plt
import os
from node import *

def process_cooja2_traces(path, tracemask):
    files = []

    # load all files and extract IPs of nodes
    for file in os.listdir(path):
        try:
            if file.startswith(tracemask) and file.index("routes"):
                continue
        except:
            files.append(file)

    nodes = pd.DataFrame(columns=['node_id', 'rank'])
    packets_node = {}

    # Load the ICMP traces
    for file in files:
        packets = pd.read_csv(path + '/' + file,
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

        packets_node[packets['node_id'][0]] = packets

    return nodes.sort_values(by=['rank','node_id']), packets_node

def separate_outliers_by_node(packets_node):
    clean_packets_node = {}

    for n in packets_node.keys():
        # Returns two DataFrames containing standard values and outliers
        mn = packets_node[n]['rtt'].mean()
        std = packets_node[n]['rtt'].std()
        upper = mn + 2 * std
        lower = mn - 2 * std

        packets_node[n]

        # Mark x(t) as outlier if mean-2*std <= x(t) <? mean+2*std
        # Maintain x(t) otherwise
        clean_packets_node[n] = packets_node[n][(packets_node[n]['rtt'] <= upper) & (packets_node[n]['rtt'] >= lower)]

    return clean_packets_node

def plot_histograms_hops_nodes(nodes, packets_node, max_x, max_y, tracemask):
    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['rank'].drop_duplicates())])

    rank_max = max(ranks)

    nodes_max = 0
    for rank in ranks:
        count = len(nodes[nodes['rank'] == rank])
        if (count > nodes_max):
            nodes_max = count

    fig = plt.figure(figsize=(4 * rank_max, 4 * nodes_max))

    # plot the histogram of RTT based on the hop
    ylabel_exists = set()
    for rank in ranks:
        count = 1
        for node in nodes[nodes['rank'] == rank].sort_values(by=['node_id'])['node_id']:
            pos = (rank - 1) + (count - 1) * rank_max + 1

            ax = plt.subplot(nodes_max, rank_max, pos)

            label = node.split(':')[4]

            packets_node[node]['rtt'].plot.kde(ax=ax, label='Node ' + label + ' KDE')
            packets_node[node]['rtt'].plot.hist(density=True, alpha=0.3, bins=50, ax=ax,
                                                label='Node ' + label + ' Hist')

            if rank == 1:
                ylabel_exists.add(count)
                ax.set_ylabel('Frequency')

            elif count not in ylabel_exists:
                ylabel_exists.add(count)
                ax.set_ylabel('Frequency')

            else:
                ax.set_ylabel('')

            if (count == 1):
                ax.set_title('Nodes at Hop ' + str(rank))

            ax.set_xlabel('Rount Trip Time (RTT) in milliseconds (ms)')
            ax.grid(axis='y')
            ax.set_xlim([0, max_x])
            ax.set_ylim([0, max_y])
            ax.legend()

            count += 1

    # ax.set_title('Distribution of the Complete Dataset Node ' + str(node) + ' at Hop ' + str(rank))
    st = fig.suptitle(tracemask[:-1], fontsize="x-large")
    plt.savefig(tracemask + 'hist.png')

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
