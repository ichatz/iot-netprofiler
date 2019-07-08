# ----------------------------------------------------------------
# IoT Netprofiler
# Licensed under The MIT License [see LICENSE for details]
# Written by Luca Maiano - https://www.linkedin.com/in/lucamaiano/
# ----------------------------------------------------------------

import pandas as pd
import numpy as np
import json
import networkx as nx
import os
import csv
from sklearn import preprocessing


def import_trace(path, file):
    # Import traces from path

    plots = set()
    with open(path + file) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        line_count = 0
        for row in csv_reader:
            if line_count == 0:
                line_count += 1
                continue
            else:
                if len(row[1].split('traces/')) > 0:
                    row[1] = row[1].split('traces/')[1]
                plots.add((path + row[1], row[2]))
    return list(plots)

def process_cooja_traces(path, tracemask, n_packets=200):
    # Process traces in cooja csv format

    files = []

    # load all files and extract IPs of nodes
    for file in os.listdir(path):
        try:
            if file.startswith(tracemask) and file.index("routes"):
                continue
        except:
            files.append(file)

    nodes = pd.DataFrame(columns=['node', 'hop'])
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
                              names=['node', 'seq', 'hop', 'rtt'],
                              engine='python').dropna().drop_duplicates()

        if len(packets) == 0:
            continue
        node = packets['node'][0]
        nodes.loc[len(nodes)] = [node, 64 - packets['hop'][0]]
        packets = packets[packets['rtt'] > 1]

        rows = []
        for i in range(1, n_packets+1):
            if i not in packets['seq'].values:
                rows.append([node, str(i), np.nan, np.nan])
        if len(rows) > 0:
            df = pd.DataFrame(np.array(rows), columns=list(packets.columns))
            packets = packets.append(df)
        packets['rtt'] = packets['rtt'].astype(float)    
        packets['seq'] = packets['seq'].astype(float)

        
        packets = packets.sort_values(by=['seq'], ascending=True, na_position='first')
        packets_node[node] = packets.reset_index(drop=True)

    return nodes.sort_values(by=['hop','node']), packets_node




def feature_extraction(nodes, packets_node, label, experiment, log_transform=False, window_size=200):
    # Input: a Dataframe nodes = node_id, rank + packets_node = {node_id: node_id, seq, hop, rtt},
    #        label that indicate the class of the experiment, the experiment_id, log_transform (a boolean 
    #        that tells if log transformation should be applied on RTT) and window_size 
    # Output: compute a dataframe containing node, experiment, transmission_time, mean, var, 
    #         packet_count, hop, min, max, loss, number_of_outliers label for each window
    
    win_stats = None
    outliers = compute_outliers_by_node(packets_node)
    for node in packets_node:
        if log_transform is True:
            packets_node[node].update(pd.DataFrame({'rtt': np.log(packets_node[node]['rtt'])}))

    	  # Features
        transmission_time = packets_node[node]['rtt'].groupby(packets_node[node]['seq'].index // window_size * window_size).sum()
        transmitted_packtes = packets_node[node]['rtt'].groupby(packets_node[node]['seq'].index // window_size * window_size).count()
        mean = packets_node[node]['rtt'].groupby(packets_node[node]['seq'].index // window_size * window_size).mean()
        var = packets_node[node]['rtt'].groupby(packets_node[node]['seq'].index // window_size * window_size).var()
        hop = int(nodes[nodes['node'] == node]['hop'])
        min_time = packets_node[node]['rtt'].groupby(packets_node[node]['seq'].index // window_size * window_size).min()
        max_time = packets_node[node]['rtt'].groupby(packets_node[node]['seq'].index // window_size * window_size).max()
        n_outliers = outliers[node]['rtt'].groupby(outliers[node]['seq'].index // window_size * window_size).count()
        loss = transmitted_packtes.copy().apply(lambda x: float(window_size) - float(x))

        
        for index in transmission_time.index:
            if win_stats is None:
                win_stats = pd.DataFrame({'node': [node],
                                       'experiment': [experiment],
                                       'tr_time': [transmission_time.loc[index]],
                                       'pckt_count': [transmitted_packtes.loc[index]],
                                       'mean': [mean.loc[index]],
                                       'var': [var.loc[index]],
                                       'hop': [hop],
                                       'min': [min_time.loc[index]],
                                       'max': [max_time.loc[index]],
                                       'loss': [loss.loc[index]],
                                       'outliers': [n_outliers.get(index, 0)],
                                       'label': [label]})
            else:
                win_stats = pd.concat([win_stats, pd.DataFrame({'node': [node],
                                       'experiment': [experiment],
                                       'tr_time': [transmission_time.loc[index]],
                                       'pckt_count': [transmitted_packtes.loc[index]],
                                       'mean': [mean.loc[index]],
                                       'var': [var.loc[index]],
                                       'hop': [hop],
                                       'min': [min_time.loc[index]],
                                       'max': [max_time.loc[index]],
                                       'loss': [loss.loc[index]],
                                       'outliers': [n_outliers.get(index, 0)],
                                       'label': [label]})])
    # Drop duplicates
    if win_stats is not None:
    	win_stats = win_stats.dropna()

    return win_stats




def feature_normalization(data, features_to_ignore=None):
    # Input: a Dataframe containing data to normalize. A set of features to ignore during normalization can
    #         be given
    # Output: normalized data

    mm_scaler = preprocessing.MinMaxScaler()
    if features_to_ignore is not None:
      to_normalize = mm_scaler.fit_transform(data.drop(features_to_ignore, axis=1))
    else:
      to_normalize = mm_scaler.fit_transform(data)
      normalized_data = pd.DataFrame(to_normalize, columns=data.columns)

    if features_to_ignore is not None:
      columns = [feature for feature in data.columns if feature not in features_to_ignore]
      normalized_data = data[features_to_ignore].join(pd.DataFrame(to_normalize, columns=columns))

    return normalized_data[list(data.columns)]




def compute_outliers_by_node(packets_node):
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
        clean_packets_node[n] = packets_node[n][(packets_node[n]['rtt'] >= upper) | (packets_node[n]['rtt'] <= lower)]

    return clean_packets_node




