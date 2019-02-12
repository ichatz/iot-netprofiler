import pandas as pd
import json
import networkx as nx
import os
import node


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
        nodes = process_iotlab_node_by_node(directory, tracefiles[i])
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


def process_iotlab_node_by_node(path, tracefile):
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
            obj_node.append(node.node(node_id, hops[node_id], nodes[node_id]))

    return obj_node


def produce_iotlab_topology(path, tracefile):
    # Load node properties to retrieve x, y location
    fo = open(path + 'nodes.out', 'r')
    data = fo.read()
    fo.close()

    # iot-lab node properties are provided in json format
    dataset = json.loads(data)
    locations = {}
    for row in dataset['items']:
        if row['archi'] == 'm3:at86rf231' and row['x'] != ' ':
            addr = int(row['network_address'].split('.')[0].split('-')[1])
            # locations[addr] = (float(row['x']), float(row['y']), float(row['z']))
            locations[addr] = (float(row['x']), float(row['y']))

    # import node IDs
    addr = pd.read_csv(path + 'addr-' + tracefile + '.cap',
                       sep='[ ;:/-]',
                       header=None,
                       usecols=[2, 23],
                       names=['node_id', 'ipv6_addr'],
                       engine='python')

    addr['node_id'] = addr['node_id'].convert_objects(convert_numeric=True)
    addr = addr.drop_duplicates(subset=['node_id'], keep="first").sort_values(by=['node_id'])
    addr.set_index('node_id')

    # Read the rank of each node
    rank = pd.read_csv(path + 'dodag-' + tracefile + '.cap',
                       sep=';|R: | \| OP:',
                       na_filter=True,
                       header=None,
                       usecols=[1, 3],
                       names=['node_id', 'rank'],
                       engine='python').dropna()

    rank.set_index('node_id')

    # Merge all data
    addr['rank'] = rank['rank'].convert_objects(convert_numeric=True)

    # build-up lookup dictionary
    ipv6 = {}
    for index, row in addr.iterrows():
        ipv6[row['ipv6_addr']] = row['node_id']

    # import RPL/dodag parents
    rpl = pd.read_csv(path + 'rpl-' + tracefile + '.cap',
                      sep='[ ;:/-]',
                      header=None,
                      usecols=[2, 11],
                      names=['node_id', 'rpl_parent'],
                      engine='python')

    rpl['node_id'] = rpl['node_id'].convert_objects(convert_numeric=True)
    rpl = rpl.drop_duplicates(subset=['node_id', 'rpl_parent'], keep="first").sort_values(by=['node_id'])
    rpl.set_index('node_id')

    # create network graph
    G = nx.DiGraph()

    for index, row in addr.iterrows():
        G.add_node(row['node_id'], addr=row['ipv6_addr'], loc=locations.get(row['node_id'], (0, 0, 0)))
        G.node[row['node_id']]['id'] = str(row['node_id'])
        G.node[row['node_id']]['addr'] = row['ipv6_addr']
        G.node[row['node_id']]['loc'] = locations.get(row['node_id'], (0, 0, 0))

        if row['rank'] == 256:
            G.node[row['node_id']]['color'] = 'red'

        elif row['rank'] == 512:
            G.node[row['node_id']]['color'] = 'green'

        elif row['rank'] == 768:
            G.node[row['node_id']]['color'] = 'blue'

        elif row['rank'] == 1024:
            G.node[row['node_id']]['color'] = 'yellow'

        else:
            G.node[row['node_id']]['color'] = 'cyan'

    for index, row in rpl.iterrows():
        if row['rpl_parent'] in ipv6 and row['node_id'] in G.nodes() and ipv6[row['rpl_parent']] in G.nodes():
            G.add_edge(row['node_id'], ipv6[row['rpl_parent']])

    return G


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
