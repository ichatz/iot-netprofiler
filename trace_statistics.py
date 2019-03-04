import numpy as np
import pandas as pd
import os
import trace_analysis
import sys

def compute_labeled_statistics_by_network(stats, feature, n_nodes):
	# Input: stats a dataframe containing the statistics of the network
	#        feature a feature to extract
	#        n_nodes the number of nodes in the network
	#Output: extract feature for each node of the network

    data = stats[['experiment',str(feature),'label']].sort_values(by=['experiment']).reset_index(drop=True)

    network = None
    experiment = None
    label = None
    nodes = []
    for index in data.index:
        # Write the experiment to a dataframe
        if experiment != data.at[index,'experiment'] and experiment != None:
            features = {'experiment': [experiment], 'label': [label]}
            for node in range(1, n_nodes+1):
                if node <= len(nodes):
                    features[node] = [nodes[node-1]]
                else:
                    features[node] = [np.float32(sys.maxsize)]

            # Create a new dataframe
            if network is None:
                network = pd.DataFrame(features)
            else:
                network = pd.concat([network, pd.DataFrame(features)])

            nodes = []
            experiment = data.at[index,'experiment']
            label = data.at[index,'label']

        # First iteration
        elif experiment == None:
            nodes = []
            experiment = data.at[index,'experiment']
            label = data.at[index,'label']

        nodes.append(data.at[index, feature])

    # Write the last experiment
    experiment = data["experiment"].iloc[-1]
    label = data["label"].iloc[-1]
    features = {'experiment': [experiment], 'label': [label]}
    for node in range(1, n_nodes+1):
        if node <= len(nodes):
            features[node] = [nodes[node-1]]
        else:
            features[node] = [np.float32(sys.maxsize)]

    # Create a new dataframe
    if network is None:
        network = pd.DataFrame(features)
    else:
        network = pd.concat([network, pd.DataFrame(features)])

    network = network.reset_index(drop=True)
    return network



def compute_window_labeled_statistics_by_network(win_stats, feature, n_nodes, window_size, n_packets=200):
	# Input: stats a dataframe containing the statistics of the network
	#        feature a feature to extract
	#        n_nodes the number of nodes in the network
	#        window_size the size of the window
	#Output: extract feature for each node of the network

    data = win_stats[['experiment','node_id',str(feature),'label']].sort_values(by=['experiment','node_id']).reset_index(drop=True)

    network = None
    experiment = None
    label = None
    nodes = {}
    for index in data.index:
        # Write the experiment to a dataframe
        if experiment != data.at[index,'experiment'] and experiment != None:
            features = {'experiment': [experiment for i in range(1,int(n_packets/window_size)+1)], 'label': [label for i in range(1,int(n_packets/window_size)+1)]}
            # For each node in the network
            for node in range(1, n_nodes+1):
            	# For each node_id
                for node_id in nodes:
                    if node_id in nodes:
                        features[node] = nodes[node_id]
                    # If some window is lost we need to add infinite values
                    if len(features[node]) < int(n_packets/window_size):
                        while len(features[node]) < int(n_packets/window_size):
                            features[node].append(np.float32(sys.maxsize))

            # Create a new dataframe
            if network is None:
                network = pd.DataFrame(features)
            else:
                network = pd.concat([network, pd.DataFrame(features)])

            nodes = {}
            experiment = data.at[index,'experiment']
            label = data.at[index,'label']

        # First iteration
        elif experiment == None:
            nodes = {}
            experiment = data.at[index,'experiment']
            label = data.at[index,'label']


        if data.at[index,'node_id'] not in nodes:
            nodes[data.at[index,'node_id']] = [data.at[index, feature]]
        else:
            nodes[data.at[index,'node_id']].append(data.at[index, feature])

    # Write the last experiment
    features = {'experiment': [experiment for i in range(1,int(n_packets/window_size)+1)], 'label': [label for i in range(1,int(n_packets/window_size)+1)]}
    # For each node in the network
    for node in range(1, n_nodes+1):
      # For each node_id
        for node_id in nodes:
            if node_id in nodes:
                features[node] = nodes[node_id]
            # If some window is lost we need to add infinite values
            if len(features[node]) < int(n_packets/window_size):
                while len(features[node]) < int(n_packets/window_size):
                    features[node].append(np.float32(sys.maxsize))

    # Create a new dataframe
    if network is None:
        network = pd.DataFrame(features)
    else:
        network = pd.concat([network, pd.DataFrame(features)])

    network = network.reset_index(drop=True)
    return network



def compute_window_labeled_statistics(nodes, packets_node, label, experiment, window_size):
    # Input: a Dataframe nodes = node_id, rank + packets_node = {node_id: node_id, seq, hop, rtt},
    #        label that indicate the class of the experiment, the experiment_id and window_size 
    # Output: compute a dataframe containing node_id, count, mean, var, std, hop, min, max, loss, label for each window
    
    win_stats = None
    outliers = trace_analysis.compute_outliers_by_node(packets_node)
    for node in packets_node:
        count = packets_node[node]['rtt'].groupby(packets_node[node]['rtt'].index // window_size * window_size).count()
        mean = packets_node[node]['rtt'].groupby(packets_node[node]['rtt'].index // window_size * window_size).mean()
        var = packets_node[node]['rtt'].groupby(packets_node[node]['rtt'].index // window_size * window_size).var()
        std = packets_node[node]['rtt'].groupby(packets_node[node]['rtt'].index // window_size * window_size).std()
        hop = int(nodes[nodes['node_id'] == node]['rank'])
        min_val = packets_node[node]['rtt'].groupby(packets_node[node]['rtt'].index // window_size * window_size).min()
        max_val = packets_node[node]['rtt'].groupby(packets_node[node]['rtt'].index // window_size * window_size).max()
        n_outliers = outliers[node]['rtt'].groupby(outliers[node]['rtt'].index // window_size * window_size).count()
        loss = count.copy().apply(lambda x: 1 - float(x)/window_size)

        
        for index in count.index:
            if win_stats is None:
                win_stats = pd.DataFrame({'node_id': [node],
                                       'experiment': [experiment],
                                       'count': [count.loc[index]],
                                       'mean': [mean.loc[index]],
                                       'var': [var.loc[index]],
                                       'std': [std.loc[index]],
                                       'hop': [hop],
                                       'min': [min_val.loc[index]],
                                       'max': [max_val.loc[index]],
                                       'loss': [loss.loc[index]],
                                       'outliers': [n_outliers.get(index, 0)],
                                       'label': [label]})
            else:
                win_stats = pd.concat([win_stats, pd.DataFrame({'node_id': [node],
                                       'experiment': [experiment],
                                       'count': [count.loc[index]],
                                       'mean': [mean.loc[index]],
                                       'var': [var.loc[index]],
                                       'std': [std.loc[index]],
                                       'hop': [hop],
                                       'min': [min_val.loc[index]],
                                       'max': [max_val.loc[index]],
                                       'loss': [loss.loc[index]],
                                       'outliers': [n_outliers.get(index, 0)],
                                       'label': [label]})])
    # Drop duplicates
    if win_stats is not None:
    	win_stats = win_stats.dropna()

    return win_stats


def compute_labeled_statistics(nodes, packets_node, label, experiment):
    # Input: a Dataframe nodes = node_id, rank + packets_node = {node_id: node_id, seq, hop, rtt}
    #        label that indicate the class of the experiment and the experiment_id
    # Output: compute a dataframe containing node_id, count, mean, var, std, hop, min, max, loss, label
    
    stats = None
    outliers = trace_analysis.compute_outliers_by_node(packets_node)
    for node in packets_node:
        count = packets_node[node]['rtt'].count()
        mean = packets_node[node]['rtt'].mean()
        var = packets_node[node]['rtt'].var()
        std = packets_node[node]['rtt'].std()
        hop = int(nodes[nodes['node_id'] == node]['rank'])
        min_val = packets_node[node]['rtt'].min()
        max_val = packets_node[node]['rtt'].max()
        n_outliers = outliers[node]['rtt'].count()
        loss = 1 - float(count)/200
        if stats is None:
            stats = pd.DataFrame({'node_id': [node],
                                   'experiment': [experiment],
                                   'count': [count],
                                   'mean': [mean],
                                   'var': [var],
                                   'std': [std],
                                   'hop': [hop],
                                   'min': [min_val],
                                   'max': [max_val],
                                   'loss': [loss],
                                   'outliers': [n_outliers],
                                   'label': [label]})
        else:
            stats = pd.concat([stats, pd.DataFrame({'node_id': [node],
                                   'experiment': [experiment],
                                   'count': [count],
                                   'mean': [mean],
                                   'var': [var],
                                   'std': [std],
                                   'hop': [hop],
                                   'min': [min_val],
                                   'max': [max_val],
                                   'loss': [loss],
                                   'outliers': [n_outliers],
                                   'label': [label]})])
    
    return stats


def tumbling_statistics_per_node(path, tracefile, window_size=10):
	# Compute a dictionary containing all the statistics from each node of the dataset

    # Read the rank of each node
    nodes = pd.read_csv(path + 'addr-' + tracefile + '.cap', 
                            sep=';|seq=| hop|time = |ms',
                            na_filter=True,
                            usecols=[1,3,5], 
                            header=None, 
                            skiprows=799,
                            names=['node_id','seq','rtt'],
                            engine='python').dropna().drop_duplicates()

    nodes = nodes.sort_values(by=['node_id','seq'], ascending=True, na_position='first')
    nodes = nodes[nodes['rtt'] >= 1]    # Removes values with RTT < 1ms

    
    d_nodes = {}   # <node_id, DataFrame containing seq and rtt columns>
    for n in nodes.index:
        if nodes['node_id'][n] in d_nodes:
            d_nodes[nodes['node_id'][n]] = d_nodes[nodes['node_id'][n]].append(pd.DataFrame({'seq': [int(nodes['seq'][n])], nodes['node_id'][n]: [nodes['rtt'][n]]}))
        else:
            d_nodes[nodes['node_id'][n]] = pd.DataFrame({'seq': [int(nodes['seq'][n])], nodes['node_id'][n]:[nodes['rtt'][n]]})


    # Generate a dataframe containing all nodes
    nodes = pd.DataFrame([seq for seq in range(1,1001)], columns=['seq']).set_index('seq')
    for node in d_nodes.keys():
        nodes = nodes.join(d_nodes[node].set_index('seq'))

    nodes = nodes[~nodes.index.duplicated(keep='first')]
    
    
    # Calculate all the statistics
    statistics = {}	# <node_id, statistics of the node>
    for node in nodes:
        stats = nodes[node].groupby(nodes[node].index // window_size).count().to_frame()
        stats = stats.rename(index=str, columns={node: "packet_loss"})
        stats["packet_loss"] = pd.to_numeric(stats["packet_loss"], downcast='float')
        for index, row in stats.iterrows():
            stats.at[index, 'packet_loss'] = row['packet_loss'] / window_size

        stats = stats.assign(mean=nodes[node].groupby(nodes[node].index // window_size).mean().values)
        stats = stats.assign(min=nodes[node].groupby(nodes[node].index // window_size).min().values)
        stats = stats.assign(max=nodes[node].groupby(nodes[node].index // window_size).max().values)
        stats = stats.assign(std=nodes[node].groupby(nodes[node].index // window_size).std().values)
        stats = stats.assign(var=nodes[node].groupby(nodes[node].index // window_size).var().values)

        statistics[node] = stats
    
    return statistics



def tumbling_packet_loss_per_hop(path, tracefile, window_size=10):
	# Compute packet loss at each hop for a given window size

    hop_nodes = trace_analysis.process_iotlab_aggregated(path, tracefile)
    
    packet_loss = pd.DataFrame()
    for hop in hop_nodes.columns.values:
        packet_loss_hop = hop_nodes[hop].groupby(hop_nodes[hop].index // window_size * window_size).dropna().count()
        packet_loss_hop[hop] = pd.to_numeric(packet_loss_hop, downcast='float')
        for index, value in packet_loss_hop.items():
            packet_loss_hop.at[index] = value / window_size

        packet_loss[hop] = packet_loss_hop


    return packet_loss
