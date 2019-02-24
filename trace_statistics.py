import numpy as np
import pandas as pd
import os
import trace_analysis



def compute_labeled_statistics(nodes, packets_node, label):
    # Input: a Dataframe nodes = node_id, rank + packets_node = {node_id: node_id, seq, hop, rtt}
    #        and label that indicate the class of the experiment
    # Output: compute a dataframe containing node_id, count, mean, var, std, hop, min, max, loss, label
    
    stats = None
    for node in packets_node:
        count = packets_node[node]['rtt'].count()
        mean = packets_node[node]['rtt'].mean()
        var = packets_node[node]['rtt'].var()
        std = packets_node[node]['rtt'].std()
        hop = int(nodes[nodes['node_id'] == node]['rank'])
        min_val = packets_node[node]['rtt'].min()
        max_val = packets_node[node]['rtt'].max()
        loss = float(count)/200
        if stats is None:
            stats = pd.DataFrame({'node_id': [node],
                                   'count': [count],
                                   'mean': [mean],
                                   'var': [var],
                                   'std': [std],
                                   'hop': [hop],
                                   'min': [min_val],
                                   'max': [max_val],
                                   'loss': [loss],
                                   'label': [label]})
        else:
            stats = pd.concat([stats, pd.DataFrame({'node_id': [node],
                                   'count': [count],
                                   'mean': [mean],
                                   'var': [var],
                                   'std': [std],
                                   'hop': [hop],
                                   'min': [min_val],
                                   'max': [max_val],
                                   'loss': [loss],
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
