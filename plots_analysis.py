from functions import *
import pandas as pd
import numpy as np
import os


def plot_histograms_hops_nodes(nodes, packets_node, max_x, max_y, tracemask):
    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['rank'].drop_duplicates())])

    rank_max = 0
    if len(ranks) > 0:
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

            if len(packets_node[node]['rtt']) > 1:
                ax = plt.subplot(nodes_max, rank_max, pos)



                label = node
                if len(label.split(':')) >= 4:
                    label = label.split(':')[4]

        
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
    st = fig.suptitle(tracemask, fontsize="x-large")
    plt.savefig('plots/' + tracemask + 'hist.png')




def plot_histograms_outliers_hops_nodes(nodes, packets_node, max_x, max_y, tracemask):
    # Plot the outliers at each hop

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['rank'].drop_duplicates())])

    rank_max = 0
    if len(ranks) > 0:
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

            if len(packets_node[node]['rtt']) > 1:
                ax = plt.subplot(nodes_max, rank_max, pos)



                label = node
                if len(label.split(':')) >= 4:
                    label = label.split(':')[4]

        
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
                    ax.set_title('Outliers at Hop ' + str(rank))

                ax.set_xlabel('Rount Trip Time (RTT) in milliseconds (ms)')
                ax.grid(axis='y')
                ax.set_xlim([0, max_x])
                ax.set_ylim([0, max_y])
                ax.legend()

            count += 1

    # ax.set_title('Distribution of the Complete Dataset Node ' + str(node) + ' at Hop ' + str(rank))
    st = fig.suptitle(tracemask, fontsize="x-large")
    plt.savefig('plots/' + tracemask + '-out-hist.png')


def plot_histograms_iqr_outliers_hops_nodes(nodes, packets_node, max_x, max_y, tracemask):
    # Print the outliers

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['rank'].drop_duplicates())])

    rank_max = 0
    if len(ranks) > 0:
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

            if len(packets_node[node]['rtt']) > 1:
                ax = plt.subplot(nodes_max, rank_max, pos)



                label = node
                if len(label.split(':')) >= 4:
                    label = label.split(':')[4]

        
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
                    ax.set_title('IQR Outliers at Hop ' + str(rank))

                ax.set_xlabel('Rount Trip Time (RTT) in milliseconds (ms)')
                ax.grid(axis='y')
                ax.set_xlim([0, max_x])
                ax.set_ylim([0, max_y])
                ax.legend()

            count += 1

    # ax.set_title('Distribution of the Complete Dataset Node ' + str(node) + ' at Hop ' + str(rank))
    st = fig.suptitle(tracemask, fontsize="x-large")
    plt.savefig('plots/' + tracemask + '-out-iqr-hist.png')


def plot_tumbling_windows_hops_nodes(nodes, packets_node, max_x, max_y, tracemask, window_size):
    # Tumbling windows

    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['rank'].drop_duplicates())])

    rank_max = 0
    if len(ranks) > 0:
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

            if len(packets_node[node]['rtt']) > 1:
                ax = plt.subplot(nodes_max, rank_max, pos)



                label = node
                if len(label.split(':')) >= 4:
                    label = label.split(':')[4]

                packets_node[node]['rtt'].groupby(packets_node[node]['rtt'].index // window_size * window_size
                    ).mean().plot(ax=ax, label='Node ' + str(node))

                if rank == 1:
                    ylabel_exists.add(count)
                    ax.set_ylabel('Mean RTT in milliseconds (ms)')

                elif count not in ylabel_exists:
                    ylabel_exists.add(count)
                    ax.set_ylabel('Mean RTT in milliseconds (ms)')

                else:
                    ax.set_ylabel('')

                if (count == 1):
                    ax.set_title('Tumbling Window of Nodes at Hop ' + str(rank))

                ax.set_xlabel('Window')
                ax.grid(axis='y')
                ax.set_xlim([0, max_x])
                ax.set_ylim([0, max_y])
                #plt.tight_layout()
                ax.legend()

            count += 1

    # ax.set_title('Distribution of the Complete Dataset Node ' + str(node) + ' at Hop ' + str(rank))
    st = fig.suptitle(tracemask, fontsize="x-large")
    plt.savefig('plots/' + tracemask + '-tumbling.png')

    

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














####################################################################################################
####################################################################################################
#######################               USING CLASS NODE              ################################
####################################################################################################
####################################################################################################

def saveFileFigures(fig,directory,namefile):
    directory=directory+"figures/"
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(directory)
    fig.savefig(directory+namefile+".pdf")   # save the figure to file
    #plt.show()


#Prints on a file the big matrix (asked by professor)
def printBigPlot(directory,data,figsize,namefile,colors,cases):
    print("Printing Big Plot for "+directory)
    fig, axs= plt.subplots(len(data),len(data[0]), figsize=figsize,sharey=True, )

    for i in range(len(data)):
        for j in range(len(data[i])):
        #print(i,j)
            ax=axs[i][j]
            d=data[i][j].pkts["rtt"]
            ax.set_ylabel("Density")
            ax.set_title("Node "+ str(data[i][j].ip) )
            ax.set_xlabel("Time (ms)")
            if not d.empty  | len(d)<2 :
                d.plot.kde(
                    ax=ax,
                    label="Case " +str(cases[i]),
                    color=colors[i]

                )


                d.hist(density=True,alpha=0.3,color=colors[i], ax=ax)

                ax.legend()
            #ax.set_xlim([-500, 8000])
    plt.tight_layout()
    saveFileFigures(fig,directory,namefile)

#Print on a file density by Hop (asked by professor)
def printDensityByHop(directory,data,hops,figsize,namefile,colors,cases):

    print("Printing Density by Hop for "+directory)
    #dataHop=hopPreparation(data)
    fig, axs= plt.subplots(len(dataHop[0]),1, figsize=(15,20),sharey=True, )
    #print(len(dataHop),len(dataHop[0]))
    for i in range(len(dataHop)):
        for j in range(len(dataHop[i])):
            #print(i,j)
            d=dataHop[i][j]['rtt']
            axs[j].set_xlabel("Time (ms)")
            axs[j].set_title("Hop "+ str(j+1))
            if not d.empty | len(d)<2 :
                d.plot.kde(
                    ax=axs[j],
                    label=cases[i],color=colors[i]
                )

                d.hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])


                axs[j].legend()

            #axs[j].set_xlim([-40, 6000])
    plt.tight_layout()
    saveFileFigures(fig,directory,namefile)

#Print on a file density by Case (asked by professor)
def printDensityByCase(directory,data,hops,figsize,namefile,colors,cases):

    print("Printing Density by case for "+directory)
    #print(len(data),len(data[0]))

    #data1=hopPreparation(data)
    dataHopT=[*zip(*hops)]

    #print(len(data1),len(data1[0]))
    #print(len(dataHopT),len(dataHopT[0]))
    fig, axs= plt.subplots(len(dataHopT[0]),1, figsize=(15,20),sharey=True, )
    for i in range(len(dataHopT)):
        for j in range(len(dataHopT[0])):
            d=dataHopT[i][j]["rtt"]
            axs[j].set_title(""+ cases[i])
            axs[j].set_xlabel("Time (ms)")
            axs[j].set_ylabel("Density")
            if not d.empty | len(d)<2 :
                d.plot.kde(
                    ax=axs[j],
                    label="Hop "+str(i),
                    color=colors[i]
                )

                d.hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])

                axs[j].legend()

    plt.tight_layout()
    #axs[j].set_xlim([-40, 6000])
    saveFileFigures(fig,directory,namefile)

#Print Density of delay without outliers in every node by Case
def densityOfDelayByCaseNoOutliers(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of delay without outliers in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            out=getStdValues(data[i][j].pkts)
            if not out.empty :
                ax=axs[j]
                out["rtt"].plot.kde(
                ax=ax,
                label=cases[i],
                     color=colors[i]
            )
                ax.set_ylabel("Density")
                out["rtt"].hist(density=True,alpha=0.3, ax=ax, color=colors[i])
                ax.set_title("Node "+ str(data[i][j].ip))
                ax.set_xlabel("Time (ms)")
                ax.legend()
    plt.tight_layout()
    saveFileFigures(fig,directory,namefile)

#Density of outliers in every node by Case
def densityOutliersByCase(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of outliers in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data),len(data[0]), figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            out=getOutliers(data[i][j].pkts)
            ax=axs[i][j]
            ax.set_ylabel("Density")
            ax.set_title("Node "+ str(data[i][j].ip))
            ax.set_xlabel("Time (ms)")
            if not out.empty | len(out)<2 :

                out["rtt"].plot.kde(
                ax=ax,
                label=cases[i],
                 color=colors[i]
            )

                out["rtt"].hist(density=True,alpha=0.3, ax=ax, color=colors[i])
                ax.legend()

    plt.tight_layout()
    saveFileFigures(fig,directory,namefile)


#Distibution of the delay divided by Node in the differents Cases
def densityOfDelayByCase(directory,data,figsize,namefile,colors,cases):
    print("Printing Density of delay in every node by Case for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            d=data[i][j].pkts["rtt"]
            axs[j].set_title("Node "+ str(data[i][j].ip))
            axs[j].set_xlabel("Time (ms)")
            axs[j].set_ylabel("Density")
            if not d.empty | len(d)<2 :


                d.plot.kde(
                    ax=axs[j],
                    label=cases[i],color=colors[i]
                )

                d.hist(density=True,alpha=0.3, ax=axs[j],color=colors[i])

                axs[j].legend()
    plt.tight_layout()
    saveFileFigures(fig,directory,namefile)


#RTT Graph
def RTTGraph(directory,data,figsize,namefile,colors,cases):
    print("Printing RTT Graph for "+directory)
    fig, axs= plt.subplots(len(data[0]),1, figsize=figsize,sharey=True, )
    for i in range(len(data)):
        for j in range(len(data[i])):
            axs[j].plot(data[i][j].pkts["seq"],data[i][j].pkts["rtt"],label=cases[i],color=colors[i]   )
            axs[j].set_title("Node "+ str(data[i][j].ip))
            axs[j].set_xlabel("Packet Number")
            axs[j].set_ylabel("Time (ms)")
            axs[j].legend()
    plt.tight_layout()
    saveFileFigures(fig,directory,namefile)