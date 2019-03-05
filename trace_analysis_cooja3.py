import pandas as pd
import json
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
        print(path, file)
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
            lpos = file.rfind('_')
            node_id = file[lpos + 1:-4]
            packets = pd.DataFrame(columns=['node_id', 'seq', 'hop', 'rtt'],
                                   data=[[node_id, 1, node_defaults[node_id], 1]])

            nodes.loc[len(nodes)] = [node_id, node_defaults[node_id]]
            packets_node[node_id] = packets

        else:
            packets['node_id'] = packets.apply(lambda row: row['node_id'][:-1], axis=1)

            nodes.loc[len(nodes)] = [packets['node_id'][0], 64 - packets['hop'][0]]

            packets = packets.sort_values(by=['node_id', 'seq'], ascending=True, na_position='first')
            packets = packets[packets['rtt'] > 1]

            packets_node[packets['node_id'][0]] = packets

    return nodes.sort_values(by=['rank', 'node_id']), packets_node


def compute_std_outliers_by_node(packets_node):
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
    clean_packets_node = {}

    for n in packets_node.keys():
        # ignore empty dataframes
        if len(packets_node[n]) == 1:
            clean_packets_node[n] = packets_node[n]

        else:
            # Mark x(t) as outlier if value Q1 < x(t) < Q3
            # Maintain x(t) otherwise
            q1 = packets_node[n]['rtt'].quantile(.25)
            q3 = packets_node[n]['rtt'].quantile(.75)

            clean_packets_node[n] = packets_node[n][
                (packets_node[n]['rtt'] <= q3) & (packets_node[n]['rtt'] >= q1)]

    return clean_packets_node


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


def produce_stats(traces, node_defaults, base_path, do_plots=True, max_x = 1000, max_y = 0.02):
    stat = {'pkt_loss': pd.DataFrame(columns=node_defaults.keys()),
            'outliers_std': pd.DataFrame(columns=node_defaults.keys()),
            'outliers_iqr': pd.DataFrame(columns=node_defaults.keys()),
            'mean_complete': pd.DataFrame(columns=node_defaults.keys(), dtype='float64'),
            'mean_std': pd.DataFrame(columns=node_defaults.keys(), dtype='float64'),
            'mean_iqr': pd.DataFrame(columns=node_defaults.keys(), dtype='float64'),
            }

    for row in traces:
        nodes, packets_node = process_cooja2_traces(row[0], row[1], node_defaults)
        clean_std = compute_std_outliers_by_node(packets_node)
        clean_iqr = compute_iqr_outliers_by_node(packets_node)

        if do_plots:
            plot_histograms_hops_nodes(nodes, packets_node, max_x, max_y, base_path + "/plots-complete/", row[1])
            plot_histograms_hops_nodes(nodes, clean_std, max_x, max_y, base_path + "/plots-std/", row[1])
            plot_histograms_hops_nodes(nodes, clean_iqr, max_x, max_y, base_path + "/plots-iqr/", row[1])

        pkt_loss = {}
        outliers_std = {}
        outliers_iqr = {}
        mean_complete = {}
        mean_std = {}
        mean_iqr = {}

        for node in packets_node.keys():
            pkt_loss[node] = len(packets_node[node]) / 200
            outliers_std[node] = (len(packets_node[node]) - len(clean_std[node])) / len(packets_node[node])
            outliers_iqr[node] = (len(packets_node[node]) - len(clean_iqr[node])) / len(packets_node[node])
            mean_complete[node] = packets_node[node]['rtt'].mean()
            mean_std[node] = clean_std[node]['rtt'].mean()
            mean_iqr[node] = clean_iqr[node]['rtt'].mean()

        stat['pkt_loss'] = stat['pkt_loss'].append(pkt_loss, ignore_index=True)
        stat['outliers_std'] = stat['outliers_std'].append(outliers_std, ignore_index=True)
        stat['outliers_iqr'] = stat['outliers_iqr'].append(outliers_iqr, ignore_index=True)
        stat['mean_complete'] = stat['mean_complete'].append(mean_complete, ignore_index=True)
        stat['mean_std'] = stat['mean_std'].append(mean_std, ignore_index=True)
        stat['mean_iqr'] = stat['mean_iqr'].append(mean_iqr, ignore_index=True)

    return stat


traces_9nodes = [
("cooja3-9nodes/traces/1bh-9", 'grid9_1bh-9_2019-02-13_19:35_'),
    ("cooja3-9nodes/traces/1gh-3", 'grid_1gh-3_2019-02-15_22:33_'),
    ("cooja3-9nodes/traces/1gh-5", 'grid_1gh-5_2019-02-15_22:09_'),
    ("cooja3-9nodes/traces/1gh-6", 'grid_1gh-6_2019-02-15_18:25_'),
    ("cooja3-9nodes/traces/1gh-7", 'grid_1gh-7_2019-02-15_21:28_'),
    ("cooja3-9nodes/traces/1gh-9", 'grid_1gh-9_2019-02-15_19:19_'),
    ("cooja3-9nodes/traces/normal", 'grid9_normal_2019-02-13_17:05_'),
    ("cooja3-9nodes/traces/normal", 'grid9_normal_2019-02-13_18:51_'),
    ("cooja3-9nodes/traces/normal", 'grid9_normal_2019-02-13_22:23_'),
    ("cooja3-9nodes/traces/1bh-3", 'grid9_1bh-3_2019-02-13_16:28_'),
    ("cooja3-9nodes/traces/1bh-3", 'grid9_1bh-3_2019-02-13_22:05_'),
    ("cooja3-9nodes/traces/1bh-5", 'grid9_1bh-5_2019-02-13_15:31_'),
    ("cooja3-9nodes/traces/1bh-5", 'grid9_1bh-5_2019-02-13_21:44_'),
    ("cooja3-9nodes/traces/1bh-6", 'grid9_1bh-6_2019-02-13_12:59_'),
    ("cooja3-9nodes/traces/1bh-6", 'grid9_1bh-6_2019-02-13_19:15_'),
    ("cooja3-9nodes/traces/1bh-7", 'grid9_1bh-7_2019-02-13_15:08_'),
    ("cooja3-9nodes/traces/1bh-7", 'grid9_1bh-7_2019-02-13_20:02_'),
    ("cooja3-9nodes/traces/1bh-9", 'grid9_1bh-9_2019-02-13_15:57_')
    ]

node_defaults_9nodes = {
    "aaaa::212:7402:2:202": 2,
    "aaaa::212:7403:3:303": 1,
    "aaaa::212:7404:4:404": 2,
    "aaaa::212:7405:5:505": 3,
    "aaaa::212:7406:6:606": 2,
    "aaaa::212:7407:7:707": 3,
    "aaaa::212:7408:8:808": 4,
    "aaaa::212:7409:9:909": 3,
    "aaaa::212:740a:a:a0a": 4}

traces_16nodes = [
    ("cooja3-16nodes/traces/normal", 'grid_normal_2019-02-19_21:23_'),
    ("cooja3-16nodes/traces/normal", 'grid_normal_2019-02-26_10:29_'),
    ("cooja3-16nodes/traces/normal", 'grid_normal_2019-02-26_10:53_'),
    ("cooja3-16nodes/traces/normal", 'grid_normal_2019-02-26_11:10_'),
    ("cooja3-16nodes/traces/normal", 'grid_normal_2019-02-26_11:48_'),
    ("cooja3-16nodes/traces/1bh-7", 'grid_1bh-7_2019-02-19_22:13_'),
    ("cooja3-16nodes/traces/1bh-9", 'grid_1bh-9_2019-02-20_00:30_'),
    ("cooja3-16nodes/traces/1gh30-7", 'grid_1gh30-7_2019-02-19_22:35_'),
    ("cooja3-16nodes/traces/1gh30-9", 'grid_1gh30-9_2019-02-20_00:12_'),
    ("cooja3-16nodes/traces/1gh50-7", 'grid_1gh50-7_2019-02-19_22:53_'),
    ("cooja3-16nodes/traces/1gh50-9", 'grid_1gh50-9_2019-02-19_23:54_'),
    ("cooja3-16nodes/traces/1gh70-7", 'grid_1gh70-7_2019-02-19_23:11_'),
    ("cooja3-16nodes/traces/1gh70-9", 'grid_1gh70-9_2019-02-19_23:34_')
]

node_defaults_16nodes = {
    "aaaa::212:7402:2:202": 2,
    "aaaa::212:7403:3:303": 1,
    "aaaa::212:7404:4:404": 1,
    "aaaa::212:7405:5:505": 2,
    "aaaa::212:7406:6:606": 3,
    "aaaa::212:7407:7:707": 2,
    "aaaa::212:7408:8:808": 2,
    "aaaa::212:7409:9:909": 2,
    "aaaa::212:740a:a:a0a": 4,
    "aaaa::212:740b:b:b0b": 3,
    "aaaa::212:740c:c:c0c": 3,
    "aaaa::212:740d:d:d0d": 4,
    "aaaa::212:740e:e:e0e": 5,
    "aaaa::212:740f:f:f0f": 4,
    "aaaa::212:7410:10:1010": 4,
    "aaaa::212:7411:11:1111": 5
}

#stats = produce_stats(traces_16nodes, node_defaults_16nodes, "cooja3-16nodes", True, 1200, 0.02)
stats = produce_stats(traces_9nodes, node_defaults_9nodes, "cooja3-9nodes", True, 1000, 0.02)

for key in stats.keys():
    file = open('cooja3-9nodes/stats-9-' + key + '.json', 'w')
    file.write(stats[key].to_json(orient='split'))
    file.close()
