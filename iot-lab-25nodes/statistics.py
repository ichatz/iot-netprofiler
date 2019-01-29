import numpy as np
import pandas as pd
import os


def statistics_per_node(dataset_path='./traces/out-2019-01JAN-28-01.cap', window_size=10):
	# Compute a dictionary containing all the statistics from each node of the dataset

    # Read the rank of each node
    nodes = pd.read_csv(dataset_path, 
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
