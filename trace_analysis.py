import pandas as pd


def process_iotlab(tracefile):
    # Read the ip of each node
    ips = pd.read_csv(tracefile,
                      sep=';|addr:|/',
                      na_filter=True,
                      usecols=[1, 3, 4],
                      header=None,
                      nrows=550,
                      names=['prefix', 'node_id', 'addr', 'ip', 'scope'],
                      engine='python').dropna()

    # extract the ip addresses
    ips = ips[ips.scope == '64  scope: global'].reset_index(drop=True).drop(['scope'], axis=1)

    # Read the rank of each node
    rank = pd.read_csv(tracefile,
                       sep=';|\t|R: | \| OP: ',
                       na_filter=True,
                       usecols=[1, 4],
                       header=None,
                       skiprows=550,
                       names=['node_id', 'rank'],
                       engine='python').dropna()

    # compute the hop of each node
    rank = rank[rank['rank'].apply(lambda x: x.isdigit())].reset_index(drop=True)

    # Merge all data
    node_ip_and_rank = pd.merge(ips, rank, how='inner').drop_duplicates()

    # Load the ICMP traces and parse the RTT
    packets = pd.read_csv('./traces/out-2019-01JAN-28-01.cap',
                        sep=';|seq=| hop|time = |ms',
                        na_filter=True,
                        usecols=[1, 3, 5],
                        header=None,
                        skiprows=799,
                        names=['node_id', 'seq', 'rtt'],
                        engine='python').dropna().drop_duplicates()

    packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
    packets = packets[packets['rtt'] > 1]

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
    nodes = pd.DataFrame([seq for seq in range(1, 1001)], columns=['seq']).set_index('seq')
    for node in d_nodes.keys():
        nodes = nodes.join(d_nodes[node].set_index('seq'))

    nodes = nodes[~nodes.index.duplicated(keep='first')]

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    rank_to_hops = sorted([int(rank) for rank in list(node_ip_and_rank['rank'].drop_duplicates())])

    hops = {}
    icmp = [x for x in range(1,len(nodes)+1)]
    for node in node_ip_and_rank.index:
        if (rank_to_hops.index(int(node_ip_and_rank['rank'][node]))+1) in hops:
            # The key should be created
            hops[rank_to_hops.index(int(node_ip_and_rank['rank'][node]))+1].append(node_ip_and_rank['node_id'][node])
        else:
            # Just append to the list of nodes
            hops[rank_to_hops.index(int(node_ip_and_rank['rank'][node]))+1] = [node_ip_and_rank['node_id'][node]]


    # Contain mean time for each distance from the root
    hop_nodes = pd.DataFrame({1: nodes[hops[1]].mean(axis=1),\
                              2: nodes[hops[2]].mean(axis=1),\
                              3: nodes[hops[3]].mean(axis=1)})

    return hop_nodes