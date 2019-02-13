import pandas as pd
import os
from node import *
import matplotlib.pyplot as plt


def process_cooja2_traces(path, tracemask, node_defaults):
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

        if len(packets) < 1:
            # Nodes affected by a black hole did not receive any packet
            node_id = file[-24:-4]
            packets = pd.DataFrame(columns=['node_id', 'seq', 'hop', 'rtt'],
                                   data=[[node_id, 1, node_defaults[node_id], 1]])

            nodes.loc[len(nodes)] = [file[-24:-4], node_defaults[node_id]]
            packets_node[file[-24:-4]] = packets

        else:
            nodes.loc[len(nodes)] = [packets['node_id'][0], 64 - packets['hop'][0]]

            packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
            packets = packets[packets['rtt'] > 1]

            packets_node[packets['node_id'][0]] = packets

    return nodes.sort_values(by=['rank', 'node_id']), packets_node


def separate_outliers_by_node(packets_node):
    clean_packets_node = {}

    for n in packets_node.keys():
        # ignore empty dataframes
        if len(packets_node[n]) == 1:
            clean_packets_node[n] = packets_node[n]

        else:
            # Returns two DataFrames containing standard values and outliers
            mn = packets_node[n]['rtt'].mean()
            std = packets_node[n]['rtt'].std()
            upper = mn + 2 * std
            lower = mn - 2 * std

            # Mark x(t) as outlier if mean-2*std <= x(t) <? mean+2*std
            # Maintain x(t) otherwise
            clean_packets_node[n] = packets_node[n][
                (packets_node[n]['rtt'] <= upper) & (packets_node[n]['rtt'] >= lower)]

    return clean_packets_node


def compute_iqr_outliers_by_node(packets_node):
    iqr = {}

    for n in packets_node.keys():
        # Returns two DataFrames containing standard values and outliers
        q1 = packets_node[n]['rtt'].quantile(.25)
        q3 = packets_node[n]['rtt'].quantile(.75)

        # Mark x(t) as outlier if mean-2*std <= x(t) <? mean+2*std
        # Maintain x(t) otherwise
        iqr[n] = packets_node[n][(packets_node[n]['rtt'] < q1) | (packets_node[n]['rtt'] > q3)]

    return iqr


def plot_histograms_hops_nodes(nodes, packets_node, max_x, max_y, path, tracemask):
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

            if len(packets_node[node]['rtt'] > 1):
                ax = plt.subplot(nodes_max, rank_max, pos)

                label = node
                if len(label.split(':')) >= 4:
                    label = label.split(':')[4]

                # make sure trace is not empty
                if len(packets_node[node]) > 1:
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
    st = fig.suptitle(tracemask[:-1].replace('_', ' '), fontsize="x-large")
    plt.savefig(path + tracemask + 'hist.png')


plots = [("cooja3-9nodes/traces/1bh-5", 'grid9_1bh-5_2019-02-13_15:31_'),
         ("cooja3-9nodes/traces/1bh-6", 'grid9_1bh-6_2019-02-13_12:59_'),
         ("cooja3-9nodes/traces/1bh-7", 'grid9_1bh-7_2019-02-13_15:08_')]

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
    nodes, packets_node = process_cooja2_traces(row[0], row[1], node_defaults)
    clean = separate_outliers_by_node(packets_node)
    plot_histograms_hops_nodes(nodes, clean, 1000, 0.02, "cooja3-9nodes/plots/", row[1])
