import pandas as pd
import numpy as np
import json
import networkx as nx
import os
from node import *
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier

 


#######################################
##### Classification Analysis #########
#######################################

def random_forests_features_selection(X_train, X_test, y_train, y_test, features):
    # Select most important features

    #Create a Gaussian Classifier
    rf_clf = RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    rf_clf.fit(X_train,y_train)
    y_pred = rf_clf.predict(X_test)

    # Feature selection
    feature_imp = pd.Series(rf_clf.feature_importances_,index=features.columns).sort_values(ascending=False)

    # Plots features with their importance score
    sns.barplot(x=feature_imp, y=feature_imp.index)
    # Add labels to your graph
    plt.xlabel('Feature Importance Score')
    plt.ylabel('Features')
    plt.title("Visualizing Important Features")
    plt.legend()
    plt.show()


def knn_test_number_of_neighbors(X_train, X_test, y_train, y_test):
    #Setup arrays to store training and test accuracies
    neighbors = np.arange(1,30)
    train_accuracy =np.empty(len(neighbors))
    test_accuracy = np.empty(len(neighbors))

    for i,k in enumerate(neighbors):
        #Setup a knn classifier with k neighbors
        knn = KNeighborsClassifier(n_neighbors=k)

        #Fit the model
        knn.fit(X_train, y_train)

        #Compute accuracy on the training set
        train_accuracy[i] = knn.score(X_train, y_train)

        #Compute accuracy on the test set
        test_accuracy[i] = knn.score(X_test, y_test)

    #Generate plot
    plt.title('kNN Varying number of neighbors')
    plt.plot(neighbors, test_accuracy, label='Testing Accuracy')
    plt.plot(neighbors, train_accuracy, label='Training accuracy')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('Accuracy')
    plt.show()




#######################################
###### Exploratory Analysis ###########
#######################################

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

        if len(packets) == 0:
            continue
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


def compute_iqr_outliers_by_node(packets_node):
    iqr = {}

    for n in packets_node.keys():
    	# Returns two DataFrames containing standard values and outliers
        q1 = packets_node[n]['rtt'].quantile(.25)
        q3 = packets_node[n]['rtt'].quantile(.75)

        packets_node[n]

        # Mark x(t) as outlier if mean-2*std <= x(t) <? mean+2*std
        # Maintain x(t) otherwise
        iqr[n] = packets_node[n][(packets_node[n]['rtt'] < q1) | (packets_node[n]['rtt'] > q3)]

    return iqr




def importCooja(directory):
    # Import trace files from the directory
    data = []
    print(directory)
    traces = directory + "traces"
    dataList = coojaJsonImporter(traces)
    for nodeList in dataList:
        data.append(node.createNodes(nodeList))
    return data


def importIOTData(directory, tracefiles):
    # Import trace files from the directory
    data = []

    # print(tracefiles)
    for i in range(len(tracefiles)):
        print("Importing " + directory + tracefiles[i])
        nodes = process_iotlab_object_node(directory, tracefiles[i])
        data.append(nodes)

    return data


def coojaJsonImporter(dir):
    # Input: directory containing the tracefiles
    # Return: computes a list of tracefiles in the directory

    dataList = []

    for file in os.listdir(dir):
        print("Importing " + file)
        with open(dir + "/" + file, 'r') as f:
            dataList.append(json.load(f))

    return dataList


def process_iotlab_object_node(path, tracefile):
    # Input: path and name of the tracefile that you want to analyze
    # Return: computes a list of Node objects

    # Read the ip of each node
    ips = pd.read_csv(path + 'addr-' + tracefile + '.cap',
                      sep=';|addr:|/',
                      na_filter=True,
                      usecols=[1, 3, 4],
                      header=None,
                      names=['prefix', 'node_id', 'addr', 'ip', 'scope'],
                      engine='python').dropna()

    # extract the ip addresses
    ips = ips[ips.scope == '64  scope: global'].reset_index(drop=True).drop(['scope'], axis=1)

    # Read the rank of each node
    rank = pd.read_csv(path + 'dodag-' + tracefile + '.cap',
                       sep=';|R: | \| OP:',
                       na_filter=True,
                       header=None,
                       usecols=[1, 3],
                       names=['node_id', 'rank'],
                       engine='python').dropna()

    # compute the hop of each node
    rank['rank'] = rank['rank'].convert_objects(convert_numeric=True)

    # Merge all data
    node_ip_and_rank = pd.merge(ips, rank, how='inner').drop_duplicates()

    # Load the ICMP traces and parse the RTT
    packets = pd.read_csv(path + 'trace-' + tracefile + '.cap',
                          sep=';|seq=| hop|time = |ms',
                          na_filter=True,
                          header=None,
                          usecols=[1, 3, 5],
                          names=['node_id', 'seq', 'rtt'],
                          engine='python').dropna().drop_duplicates()

    packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
    packets = packets[packets['rtt'] > 1]

    max_seq = packets['seq'].max()

    # Compute the 2 dimensional array
    nodes = {}  # <node_id, DataFrame containing seq and rtt columns>
    for n in packets.index:
        if packets['node_id'][n] in nodes:
            nodes[packets['node_id'][n]] = nodes[packets['node_id'][n]].append(
                pd.DataFrame({'seq': [int(packets['seq'][n])], 'rtt': [packets['rtt'][n]]}))
        else:
            nodes[packets['node_id'][n]] = pd.DataFrame(
                {'seq': [int(packets['seq'][n])], 'rtt': [packets['rtt'][n]]})

    # We maintain just the first 100 ICMP messages
    for node_id in nodes:
        nodes[node_id] = nodes[node_id].loc[nodes[node_id]['seq'] <= 100].reset_index().drop(columns=['index'])

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    rank_to_hops = sorted([int(rank) for rank in list(node_ip_and_rank['rank'].drop_duplicates())])

    # remove root (if it exists)
    if 256 in rank_to_hops:
        rank_to_hops.remove(256)

    hops = {}
    for node_id in node_ip_and_rank.index:
        if not node_ip_and_rank['node_id'][node_id] in nodes.keys():
            continue

        if not node_ip_and_rank['rank'][node_id] in rank_to_hops:
            continue

        if (node_ip_and_rank['node_id'][node_id]) not in hops:
            # Just append to the list of nodes
            hops[node_ip_and_rank['node_id'][node_id]] = rank_to_hops.index(node_ip_and_rank['rank'][node_id]) + 1

    obj_node = []
    # Create a list of Node objects
    for node_id in nodes:
        if (node_id in hops):
            obj_node.append(node(node_id, hops[node_id], nodes[node_id]))

    return obj_node


def process_iotlab_aggregated(path, tracefile):
    # Read the ip of each node
    ips = pd.read_csv(path + 'addr-' + tracefile + '.cap',
                      sep=';|addr:|/',
                      na_filter=True,
                      usecols=[1, 3, 4],
                      header=None,
                      names=['prefix', 'node_id', 'addr', 'ip', 'scope'],
                      engine='python').dropna()

    # extract the ip addresses
    ips = ips[ips.scope == '64  scope: global'].reset_index(drop=True).drop(['scope'], axis=1)

    # Read the rank of each node
    rank = pd.read_csv(path + 'dodag-' + tracefile + '.cap',
                       sep=';|R: | \| OP:',
                       na_filter=True,
                       header=None,
                       usecols=[1, 3],
                       names=['node_id', 'rank'],
                       engine='python').dropna()

    # compute the hop of each node
    rank['rank'] = rank['rank'].convert_objects(convert_numeric=True)

    # Merge all data
    node_ip_and_rank = pd.merge(ips, rank, how='inner').drop_duplicates()

    # Load the ICMP traces and parse the RTT
    packets = pd.read_csv(path + 'trace-' + tracefile + '.cap',
                          sep=';|seq=| hop|time = |ms',
                          na_filter=True,
                          header=None,
                          usecols=[1, 3, 5],
                          names=['node_id', 'seq', 'rtt'],
                          engine='python').dropna().drop_duplicates()

    packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
    packets = packets[packets['rtt'] > 1]

    max_seq = packets['seq'].max()

    # Compute the 2 dimensional array
    d_nodes = {}  # <node_id, DataFrame containing seq and rtt columns>
    for n in packets.index:
        if packets['node_id'][n] in d_nodes:
            d_nodes[packets['node_id'][n]] = d_nodes[packets['node_id'][n]].append(
                pd.DataFrame({'seq': [int(packets['seq'][n])], packets['node_id'][n]: [packets['rtt'][n]]}))
        else:
            d_nodes[packets['node_id'][n]] = pd.DataFrame(
                {'seq': [int(packets['seq'][n])], packets['node_id'][n]: [packets['rtt'][n]]})

    # create the 2 dimensional array
    nodes = pd.DataFrame([seq for seq in range(1, max_seq + 1)], columns=['seq']).set_index('seq')
    for node in d_nodes.keys():
        nodes = nodes.join(d_nodes[node].set_index('seq'))

    nodes = nodes[~nodes.index.duplicated(keep='first')]

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    rank_to_hops = sorted([int(rank) for rank in list(node_ip_and_rank['rank'].drop_duplicates())])

    # remove root (if it exists)
    if 256 in rank_to_hops:
        rank_to_hops.remove(256)

    hops = {}
    for node in node_ip_and_rank.index:
        if not node_ip_and_rank['node_id'][node] in d_nodes.keys():
            continue

        if not node_ip_and_rank['rank'][node] in rank_to_hops:
            continue

        if (rank_to_hops.index(node_ip_and_rank['rank'][node]) + 1) in hops:
            # The key should be created
            hops[rank_to_hops.index(node_ip_and_rank['rank'][node]) + 1].append(node_ip_and_rank['node_id'][node])
        else:
            # Just append to the list of nodes
            hops[rank_to_hops.index(node_ip_and_rank['rank'][node]) + 1] = [node_ip_and_rank['node_id'][node]]

    # construct hop based statistics
    hop_nodes = pd.DataFrame({1: nodes[hops[1]].mean(axis=1)})

    for hop in range(2, max(list(hops.keys())) + 1):
        hop_nodes[hop] = nodes[hops[hop]].mean(axis=1)

    return hop_nodes


def separate_outliers(hop_nodes):
    window_size = 10
    std_values = pd.DataFrame(columns=[hop for hop in range(1, max(
        list(hop_nodes.keys())) + 1)])  # Maintain x(t) if mean-2*std <= x(t) <? mean+2*std
    outliers = pd.DataFrame(
        columns=[hop for hop in range(1, max(list(hop_nodes.keys())) + 1)])  # Maintain x(t) otherwise

    for h in hop_nodes.keys():
        # Returns two DataFrames containing standard values and outliers
        mn = hop_nodes[h].mean()
        std = hop_nodes[h].std()
        std_window = pd.Series([])  # Standard values
        out_window = pd.Series([])  # Outliers

        for window in (hop_nodes[h].groupby(hop_nodes[h].index // window_size * window_size)):
            std_curr = []
            out_curr = []
            for x in window[1]:
                if mn - 2 * std <= x and x <= mn + 2 * std:
                    std_curr.append(x)
                    out_curr.append(None)
                else:
                    std_curr.append(None)
                    out_curr.append(x)

            std_window = std_window.append(pd.Series(std_curr))
            out_window = out_window.append(pd.Series(out_curr))

        std_values[h] = std_window
        outliers[h] = out_window

    std_values = std_values.reset_index().drop(columns=['index'])
    std_values.fillna(value=pd.np.nan, inplace=True)
    outliers = outliers.reset_index().drop(columns=['index'])
    outliers.fillna(value=pd.np.nan, inplace=True)

    return std_values, outliers


def process_iotlab_node_by_node2(path, tracefile):
    # Returns a dataframe containing nodes and hops
    # and a dictionary <node_id, dataframe containing node_id,hop,seq,rtt

    # Read the rank of each node
    ranks = pd.read_csv(path + 'dodag-' + tracefile + '.cap',
                           sep=';|R: | \| OP:',
                           na_filter=True,
                           header=None,
                           usecols=[1, 3],
                           names=['node_id', 'rank'],
                           engine='python').dropna()

    # Load the ICMP traces and parse the RTT
    packets = pd.read_csv(path + 'trace-' + tracefile + '.cap',
                          sep=';|seq=| hop|time = |ms',
                          na_filter=True,
                          header=None,
                          usecols=[1, 3, 5],
                          names=['node_id', 'seq', 'rtt'],
                          engine='python').dropna().drop_duplicates()

    packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
    packets = packets[packets['rtt'] > 1]

    max_seq = packets['seq'].max()
    ranks = ranks.drop(ranks[ranks['rank'] == 256].index)

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    rank_to_hops = sorted(set([int(rank) for rank in list(ranks['rank'])]))

    # Compute the 2 dimensional array
    d_packets = {}  # <node_id, DataFrame containing seq and rtt columns>
    for n in packets.index:
        if packets['node_id'][n] in d_packets:
            d_packets[packets['node_id'][n]] = d_packets[packets['node_id'][n]].append(
                pd.DataFrame({'node_id': packets['node_id'][n], 'seq': [int(packets['seq'][n])], 'rtt': [packets['rtt'][n]]})).reset_index(drop=True)
        else:
            d_packets[packets['node_id'][n]] = pd.DataFrame(
                {'node_id': packets['node_id'][n], 'seq': [int(packets['seq'][n])], 'rtt': [packets['rtt'][n]]})

    remove = list(set(d_packets.keys()).difference(set(ranks['node_id']).intersection(set(d_packets.keys())))) # a list of nodes to be removed
    # Append the hop column to each dataframe
    for node in ranks.index:
        if ranks['node_id'][node] not in d_packets:
            remove.append(ranks['node_id'][node])
            continue

        d_packets[ranks['node_id'][node]]['hop'] = rank_to_hops.index(ranks['rank'][node]) + 1
        d_packets[ranks['node_id'][node]] = d_packets[ranks['node_id'][node]].loc[d_packets[ranks['node_id'][node]]['seq'] <= 200].reset_index(drop=True)

        # If the node was unavailable during the first 100 ICMP messages it should be removed
        if len(d_packets[ranks['node_id'][node]]) == 0:
            remove.append(ranks['node_id'][node])

    # Remove empty nodes
    for node in set(remove):
        if node in d_packets:
            del d_packets[node]


    # Compute a new DataFrame containing node_id and rank
    nodes = {}
    for node in d_packets:
        if 'node_id' not in nodes:
            nodes['node_id'] = [d_packets[node]['node_id'].iloc[0]]
        if 'rank' not in nodes:
            nodes['rank'] = [d_packets[node]['hop'].iloc[0]]
        else:
            nodes['node_id'].append(d_packets[node]['node_id'].iloc[0])
            nodes['rank'].append(d_packets[node]['hop'].iloc[0])

    return pd.DataFrame(nodes).sort_values(by=['rank','node_id']), d_packets



def process_iotlab_node_by_node(path, tracefile):
    # Read the ip of each node
    ips = pd.read_csv(path + 'addr-' + tracefile + '.cap',
                      sep=';|addr:|/',
                      na_filter=True,
                      usecols=[1, 3, 4],
                      header=None,
                      names=['prefix', 'node_id', 'addr', 'ip', 'scope'],
                      engine='python').dropna()

    # extract the ip addresses
    ips = ips[ips.scope == '64  scope: global'].reset_index(drop=True).drop(['scope'], axis=1)

    # Read the rank of each node
    rank = pd.read_csv(path + 'dodag-' + tracefile + '.cap',
                       sep=';|R: | \| OP:',
                       na_filter=True,
                       header=None,
                       usecols=[1, 3],
                       names=['node_id', 'rank'],
                       engine='python').dropna()

    # compute the hop of each node
    rank['rank'] = rank['rank'].convert_objects(convert_numeric=True)

    # Merge all data
    node_ip_and_rank = pd.merge(ips, rank, how='inner').drop_duplicates()

    # Load the ICMP traces and parse the RTT
    packets = pd.read_csv(path + 'trace-' + tracefile + '.cap',
                          sep=';|seq=| hop|time = |ms',
                          na_filter=True,
                          header=None,
                          usecols=[1, 3, 5],
                          names=['node_id', 'seq', 'rtt'],
                          engine='python').dropna().drop_duplicates()

    packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
    packets = packets[packets['rtt'] > 1]

    max_seq = packets['seq'].max()

    # Compute the 2 dimensional array
    d_nodes = {}  # <node_id, DataFrame containing seq and rtt columns>
    for n in packets.index:
        if packets['node_id'][n] in d_nodes:
            d_nodes[packets['node_id'][n]] = d_nodes[packets['node_id'][n]].append(
                pd.DataFrame({'seq': [int(packets['seq'][n])], packets['node_id'][n]: [packets['rtt'][n]]}))
        else:
            d_nodes[packets['node_id'][n]] = pd.DataFrame(
                {'seq': [int(packets['seq'][n])], packets['node_id'][n]: [packets['rtt'][n]]})

    # create the 2 dimensional array
    nodes = pd.DataFrame([seq for seq in range(1, max_seq + 1)], columns=['seq']).set_index('seq')
    for node in d_nodes.keys():
        nodes = nodes.join(d_nodes[node].set_index('seq'))

    nodes = nodes[~nodes.index.duplicated(keep='first')]

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    rank_to_hops = sorted([int(rank) for rank in list(node_ip_and_rank['rank'].drop_duplicates())])

    # remove root (if it exists)
    if 256 in rank_to_hops:
        rank_to_hops.remove(256)

    hops = {}
    for node in node_ip_and_rank.index:
        if not node_ip_and_rank['node_id'][node] in d_nodes.keys():
            continue

        if not node_ip_and_rank['rank'][node] in rank_to_hops:
            continue

        if (rank_to_hops.index(node_ip_and_rank['rank'][node]) + 1) in hops:
            # The key should be created
            hops[rank_to_hops.index(node_ip_and_rank['rank'][node]) + 1].append(node_ip_and_rank['node_id'][node])
        else:
            # Just append to the list of nodes
            hops[rank_to_hops.index(node_ip_and_rank['rank'][node]) + 1] = [node_ip_and_rank['node_id'][node]]

    return nodes, hops


def separate_outliers_node_by_node(nodes):
    window_size = 10
    std_values = pd.DataFrame(columns=[node for node in
                                       list(nodes.keys())])  # Maintain x(t) if mean-2*std <= x(t) <? mean+2*std
    outliers = pd.DataFrame(
        columns=[node for node in list(nodes.keys())])  # Maintain x(t) otherwise

    for n in nodes.keys():
        # Returns two DataFrames containing standard values and outliers
        mn = nodes[n].mean()
        std = nodes[n].std()
        std_window = pd.Series([])  # Standard values
        out_window = pd.Series([])  # Outliers

        for window in (nodes[n].groupby(nodes[n].index // window_size * window_size)):
            std_curr = []
            out_curr = []
            for x in window[1]:
                if mn - 2 * std <= x and x <= mn + 2 * std:
                    std_curr.append(x)
                    out_curr.append(None)
                else:
                    std_curr.append(None)
                    out_curr.append(x)

            std_window = std_window.append(pd.Series(std_curr))
            out_window = out_window.append(pd.Series(out_curr))

        std_values[n] = std_window
        outliers[n] = out_window

    std_values = std_values.reset_index().drop(columns=['index'])
    std_values.fillna(value=pd.np.nan, inplace=True)
    outliers = outliers.reset_index().drop(columns=['index'])
    outliers.fillna(value=pd.np.nan, inplace=True)

    return std_values, outliers


def processed_data_for_kmeans(path, tracefile):
    # Read the ip of each node
    ips = pd.read_csv(path + 'addr-' + tracefile + '.cap',
                      sep=';|addr:|/',
                      na_filter=True,
                      usecols=[1, 3, 4],
                      header=None,
                      names=['prefix', 'node_id', 'addr', 'ip', 'scope'],
                      engine='python').dropna()

    # extract the ip addresses
    ips = ips[ips.scope == '64  scope: global'].reset_index(drop=True).drop(['scope'], axis=1)

    # Read the rank of each node
    rank = pd.read_csv(path + 'dodag-' + tracefile + '.cap',
                       sep=';|R: | \| OP:',
                       na_filter=True,
                       header=None,
                       usecols=[1, 3],
                       names=['node_id', 'rank'],
                       engine='python').dropna()

    # compute the hop of each node
    rank['rank'] = rank['rank'].convert_objects(convert_numeric=True)

    # Merge all data
    node_ip_and_rank = pd.merge(ips, rank, how='inner').drop_duplicates()

    # Load the ICMP traces and parse the RTT
    packets = pd.read_csv(path + 'trace-' + tracefile + '.cap',
                          sep=';|seq=| hop|time = |ms',
                          na_filter=True,
                          header=None,
                          usecols=[1, 3, 5],
                          names=['node_id', 'seq', 'rtt'],
                          engine='python').dropna().drop_duplicates()

    packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
    packets = packets[packets['rtt'] > 1]

    max_seq = packets['seq'].max()

    # Compute the 2 dimensional array
    d_nodes = {}  # <node_id, DataFrame containing seq and rtt columns>
    for n in packets.index:
        if packets['node_id'][n] in d_nodes:
            d_nodes[packets['node_id'][n]] = d_nodes[packets['node_id'][n]].append(
                pd.DataFrame({'seq': [int(packets['seq'][n])], packets['node_id'][n]: [packets['rtt'][n]]}))
        else:
            d_nodes[packets['node_id'][n]] = pd.DataFrame(
                {'seq': [int(packets['seq'][n])], packets['node_id'][n]: [packets['rtt'][n]]})

    # create the 2 dimensional array
    nodes = pd.DataFrame([seq for seq in range(1, max_seq + 1)], columns=['seq']).set_index('seq')
    for node in d_nodes.keys():
        nodes = nodes.join(d_nodes[node].set_index('seq'))

    nodes = nodes[~nodes.index.duplicated(keep='first')]

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    rank_to_hops = sorted([int(rank) for rank in list(node_ip_and_rank['rank'].drop_duplicates())])

    # remove root (if it exists)
    if 256 in rank_to_hops:
        rank_to_hops.remove(256)

    hops = {}
    for node in node_ip_and_rank.index:
        if not node_ip_and_rank['node_id'][node] in d_nodes.keys():
            continue

        if not node_ip_and_rank['rank'][node] in rank_to_hops:
            continue

        hops[node_ip_and_rank['node_id'][node]] = rank_to_hops.index(node_ip_and_rank['rank'][node]) + 1

    node_dataframe = {}
    for node in nodes.keys():
        node_dataframe[node] = nodes[node].to_frame().rename(index=str, columns={node: 'rtt'})
        node_dataframe[node]['hop'] = hops[node]
        node_dataframe[node]['seq'] = node_dataframe[node].index

    return node_dataframe
