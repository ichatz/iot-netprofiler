# ----------------------------------------------------------------
# IoT Netprofiler
# Licensed under The MIT License [see LICENSE for details]
# Written by Luca Maiano - https://www.linkedin.com/in/lucamaiano/
# ----------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns

from mpl_toolkits.mplot3d import Axes3D
from os import listdir
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier


#######################################
#######   FEATURE SELECTION   #########
#######################################

def random_forests_features_selection(data, tracemask):
    fig, ax = plt.subplots(figsize=(10,6)) 

    # separate features from target values
    features = data.drop(columns=['label'])
    target = data['label'].values

    # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=3)

    #Create a Gaussian Classifier
    rf_clf = RandomForestClassifier(n_estimators=100)

    #Train the model using the training sets y_pred=clf.predict(X_test)
    rf_clf.fit(X_train,y_train)
    y_pred = rf_clf.predict(X_test)

    # Feature selection
    feature_imp = pd.Series(rf_clf.feature_importances_,index=features.columns).sort_values(ascending=False)

    # Plots features with their importance score
    feature_imp.plot.barh()

    # Add labels to your graph
    plt.xlabel('Importance Score')
    plt.ylabel('Features')
    
    st = fig.suptitle('RandomForestClassifier\'s Feature Importance', fontsize="x-large")

    try:
        if 'figures' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/figures')
        if 'feature_importance' not in listdir(os.getcwd()+'/data/figures/'):
            os.makedirs('data/figures/feature_importance')
        if len(tracemask.split('_')) > 3:
            tracemask = str(tracemask.split('_')[0]) + str(tracemask.split('_')[1]) + str(tracemask.split('_')[3])
        
        plt.savefig('data/figures/feature_importance/' + tracemask + '_feature_importance.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + tracemask + '_feature_importance.png')


def knn_test_number_of_neighbors(trace_stats, max_neighbors):
    # Test the accuracy of KNN for different number of neighbors
    max_rows = len(trace_stats)/2
    max_cols = 2
    pos = 1
    fig = plt.figure(figsize=(6*max_rows , 4*max_cols))
    fig.subplots_adjust(hspace=0.4, wspace=0.4)

    # Select trace
    for trace_size in trace_stats:
        ax = plt.subplot(max_rows, max_cols, pos)
        trace = trace_stats[trace_size]

        # separate features from target values
        features = trace.drop(columns=['node', 'experiment', 'label'])
        target = trace['label'].values

        # split dataset into train and test data
        X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)

	    #Setup arrays to store training and test accuracies
        neighbors = np.arange(1, max_neighbors)
        train_accuracy = np.empty(len(neighbors))
        test_accuracy = np.empty(len(neighbors))

        for i, k in enumerate(neighbors):
            #Setup a knn classifier with k neighbors
            knn = KNeighborsClassifier(n_neighbors=k)

            #Fit the model
            knn.fit(X_train, y_train)

            #Compute accuracy on the training set
            train_accuracy[i] = knn.score(X_train, y_train)

            #Compute accuracy on the test set
            test_accuracy[i] = knn.score(X_test, y_test)

        #Generate plot
        plt.title('Window Size {}'.format(trace_size))
        pd.DataFrame({'Testing Accuracy':test_accuracy}).plot(label='Testing Accuracy', ax=ax)
        pd.DataFrame({'Training Accuracy':train_accuracy}).plot(label='Training Accuracy', ax=ax)
        plt.xlabel('Number of neighbors')
        plt.ylabel('Accuracy')
        pos += 1

    st = fig.suptitle('KNN Varying number of neighbors', fontsize="x-large")
    plt.show()


def visualize_loss_function(model, history):
    # summarize history for loss
    fig = plt.figure(figsize=(10,6))
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'Validation'], loc='upper left')

    plt.title('Training and Validation Loss functions', fontsize="x-large")
    #plt.show()

    try:
        if 'figures' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/figures')
        if 'loss' not in listdir(os.getcwd()+'/data/figures/'):
            os.makedirs('data/figures/loss')
        
        plt.savefig('data/figures/loss/' + model + '_loss.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + model + '_loss.png')


def plot_3d_points(X0, X1, X2, y, plot_name, centroids=None):
    fig = plt.figure(figsize=(10,6))
    ax = Axes3D(fig)
    colors = {0: 'red', 1: 'green', 2: 'blue', 3: 'cyan', 4: 'magenta', 5: 'yellow', 6: 'black'}

    for class_i in set(y):
        indexes_i = y.where(y == class_i).dropna().index.values
        ax.scatter(X0.ix[indexes_i], X1.ix[indexes_i], X2.ix[indexes_i], c=colors[class_i], label='Class ' +str(class_i), alpha=0.8)
    
    ax.legend()

    if centroids is None:
        st = fig.suptitle('PCA Transformation ' + str(len(set(y))) + ' Classes', fontsize="x-large")
    else:
        ax.scatter(centroids[:, 0], centroids[:, 1], centroids[:, 2], marker='*', s=2000)
        st = fig.suptitle('K-Means Clusters', fontsize="x-large")

    try:
        if 'figures' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/figures')
        if '3d_plots' not in listdir(os.getcwd()+'/data/figures/'):
            os.makedirs('data/figures/3d_plots')
        
        plt.savefig('data/figures/3d_plots/' + plot_name + '_3d.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + plot_name + '_3d.png')




#######################################
#######   DATA EXPLORATION   ##########
#######################################

def plot_correlation_matrix(data, tracemask):
    # Takes in input a dataframe and plots the corresponding correlation matrix

    fig, ax = plt.subplots(figsize=(10,10)) 
    corr = data.corr()
    ax = sns.heatmap(
        corr, 
        vmin=-1, vmax=1, center=0,
        cmap=sns.diverging_palette(20, 220, n=250),
        square=True
    )
    ax.set_xticklabels(
        ax.get_xticklabels(),
        rotation=45,
        horizontalalignment='right'
    )
    st = fig.suptitle('Correlation Matrix', fontsize="x-large")
    try:
        if 'figures' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/figures')
        if 'correlation' not in listdir(os.getcwd()+'/data/figures/'):
            os.makedirs('data/figures/correlation')
        if len(tracemask.split('_')) > 3:
            tracemask = str(tracemask.split('_')[0]) + str(tracemask.split('_')[1]) + str(tracemask.split('_')[3])
        
        plt.savefig('data/figures/correlation/' + tracemask + '_correlation.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + tracemask + '_correlation.png')



def rtt_distrubution_of_nodes(nodes, packets_node, max_x, max_y, topology, tracemask):
    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['hop'].drop_duplicates())])

    rank_max = 0
    if len(ranks) > 0:
        rank_max = max(ranks)

    nodes_max = 0
    for rank in ranks:
        count = len(nodes[nodes['hop'] == rank])
        if (count > nodes_max):
            nodes_max = count

    fig = plt.figure(figsize=(4 * rank_max, 4 * nodes_max))


    # plot the histogram of RTT based on the hop
    ylabel_exists = set()
    for rank in ranks:
        count = 1
        for node in nodes[nodes['hop'] == rank].sort_values(by=['node'])['node']:
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

    st = fig.suptitle(topology+'-'+tracemask, fontsize="x-large")
    try:
        if 'figures' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/figures')
        if 'rtt_distributions' not in listdir(os.getcwd()+'/data/figures/'):
            os.makedirs('data/figures/rtt_distributions')
        if len(tracemask.split('_')) > 3:
            tracemask = str(tracemask.split('_')[0]) + str(tracemask.split('_')[1]) + str(tracemask.split('_')[3])
        
        plt.savefig('data/figures/rtt_distributions/' + topology + '-' + tracemask + '.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + topology + '-' + tracemask + '.png')



def rtt_autocorrelation_of_nodes(nodes, packets_node, max_x, min_y,max_y, topology, tracemask):
    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['hop'].drop_duplicates())])

    rank_max = 0
    if len(ranks) > 0:
        rank_max = max(ranks)

    nodes_max = 0
    for rank in ranks:
        count = len(nodes[nodes['hop'] == rank])
        if (count > nodes_max):
            nodes_max = count

    fig = plt.figure(figsize=(4 * rank_max, 4 * nodes_max))


    # plot the histogram of RTT based on the hop
    ylabel_exists = set()
    for rank in ranks:
        count = 1
        for node in nodes[nodes['hop'] == rank].sort_values(by=['node'])['node']:
            pos = (rank - 1) + (count - 1) * rank_max + 1

            if len(packets_node[node]['rtt']) > 1:
                ax = plt.subplot(nodes_max, rank_max, pos)



                label = node
                if len(label.split(':')) >= 4:
                    label = label.split(':')[4]

                autocorrelation = []
                for i in range(1,int(len(packets_node[node]['rtt'])/2+1)):
                    autocorrelation.append(packets_node[node]['rtt'].autocorr(i))

                ax.bar(range(1,101),autocorrelation, width=0.5, label='Autocorrelation Node ' + label)

                if rank == 1:
                    ylabel_exists.add(count)
                    ax.set_ylabel('ACF')

                elif count not in ylabel_exists:
                    ylabel_exists.add(count)
                    ax.set_ylabel('ACF')

                else:
                    ax.set_ylabel('')

                if (count == 1):
                    ax.set_title('Nodes at Hop ' + str(rank))

                ax.set_xlabel('Lags')
                ax.grid(axis='y')
                ax.set_xlim([0, max_x])
                ax.set_ylim([min_y, max_y])
                ax.legend()

            count += 1

    st = fig.suptitle(topology+'-'+tracemask, fontsize="x-large")
    try:
        if 'figures' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/figures')
        if 'autocorrelations' not in listdir(os.getcwd()+'/data/figures/'):
            os.makedirs('data/figures/autocorrelations')
        if len(tracemask.split('_')) > 3:
            tracemask = str(tracemask.split('_')[0]) + str(tracemask.split('_')[1]) + str(tracemask.split('_')[3])
        
        plt.savefig('data/figures/autocorrelations/' + topology + '-' + tracemask + '.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + topology + '-' + tracemask + '.png')



def rtt_exponential_smoothing_of_nodes(nodes, packets_node, alpha, max_x, max_y, topology, tracemask):
    # Each node communicates with the root of the DODAG through a certain number of hops.
    # The network was configured in order to have three nodes communicating directly with the root.
    ranks = sorted([int(rank) for rank in list(nodes['hop'].drop_duplicates())])

    rank_max = 0
    if len(ranks) > 0:
        rank_max = max(ranks)

    nodes_max = 0
    for rank in ranks:
        count = len(nodes[nodes['hop'] == rank])
        if (count > nodes_max):
            nodes_max = count

    fig = plt.figure(figsize=(4 * rank_max, 4 * nodes_max))


    # plot the histogram of RTT based on the hop
    ylabel_exists = set()
    for rank in ranks:
        count = 1
        for node in nodes[nodes['hop'] == rank].sort_values(by=['node'])['node']:
            pos = (rank - 1) + (count - 1) * rank_max + 1

            if len(packets_node[node]['rtt']) > 1:
                ax = plt.subplot(nodes_max, rank_max, pos)

                label = node
                if len(label.split(':')) >= 4:
                    label = label.split(':')[4]

                packets_node[node]['rtt'].plot(ax=ax, label='Node ' + label + '\'s RTT')
                packets_node[node]['rtt'].ewm(alpha = 0.8).mean().plot(ax=ax, label='Alpha ' + str(alpha))

                if rank == 1:
                    ylabel_exists.add(count)
                    ax.set_ylabel('RTT')

                elif count not in ylabel_exists:
                    ylabel_exists.add(count)
                    ax.set_ylabel('RTT')

                else:
                    ax.set_ylabel('')

                if (count == 1):
                    ax.set_title('Nodes at Hop ' + str(rank))

                ax.set_xlabel('ICMP Messages')
                ax.grid(axis='y')
                ax.set_xlim([0, max_x])
                ax.set_ylim([0, max_y])
                ax.legend()

            count += 1

    st = fig.suptitle(topology+'-'+tracemask, fontsize="x-large")
    try:
        if 'figures' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/figures')
        if 'exponential_smoothing' not in listdir(os.getcwd()+'/data/figures/'):
            os.makedirs('data/figures/exponential_smoothing')
        if len(tracemask.split('_')) > 3:
            tracemask = str(tracemask.split('_')[0]) + str(tracemask.split('_')[1]) + str(tracemask.split('_')[3])
        
        plt.savefig('data/figures/exponential_smoothing/' + topology + '-' + tracemask + '.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + topology + '-' + tracemask + '.png')





















####################################
#########     OLD API      #########
####################################


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
        for node in nodes[nodes['rank'] == rank].sort_values(by=['node'])['node']:
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
    try:
        if 'plots' not in listdir(os.getcwd()):
            os.makedirs('plots')
        plt.savefig('plots/' + tracemask + '-out-hist.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + tracemask + '-out-hist.png')


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
        for node in nodes[nodes['rank'] == rank].sort_values(by=['node'])['node']:
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
    try:
        if 'plots' not in listdir(os.getcwd()):
            os.makedirs('plots')
        plt.savefig('plots/' + tracemask + '-out-iqr-hist.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + tracemask + '-out-iqr-hist.png')


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
        for node in nodes[nodes['rank'] == rank].sort_values(by=['node'])['node']:
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
                    ax.set_title('Window of size ' + str(window_size) + ' of Nodes at Hop ' + str(rank))

                ax.set_xlabel('Window')
                ax.grid(axis='y')
                ax.set_xlim([0, max_x])
                ax.set_ylim([0, max_y])
                #plt.tight_layout()
                ax.legend()

            count += 1

    # ax.set_title('Distribution of the Complete Dataset Node ' + str(node) + ' at Hop ' + str(rank))
    st = fig.suptitle(tracemask, fontsize="x-large")
    try:
        if 'plots' not in listdir(os.getcwd()):
            os.makedirs('plots')
        plt.savefig('plots/' + tracemask + '-tumbling.png')
    except:
        print("Invalid IHDR data. Cannot save the image " + tracemask + '-tumbling.png')


def visualize_topology(path, tracefile):
    # create network graph
    G = nx.DiGraph()

    for index, row in addr.iterrows():
        G.add_node(row['node'], addr=row['ipv6_addr'], loc=locations.get(row['node'], (0, 0, 0)))
        G.node[row['node']]['id'] = str(row['node'])
        G.node[row['node']]['addr'] = row['ipv6_addr']
        G.node[row['node']]['loc'] = locations.get(row['node'], (0, 0, 0))

        if row['rank'] == 256:
            G.node[row['node']]['color'] = 'red'

        elif row['rank'] == 512:
            G.node[row['node']]['color'] = 'green'

        elif row['rank'] == 768:
            G.node[row['node']]['color'] = 'blue'

        elif row['rank'] == 1024:
            G.node[row['node']]['color'] = 'yellow'

        else:
            G.node[row['node']]['color'] = 'cyan'

    for index, row in rpl.iterrows():
        if row['rpl_parent'] in ipv6 and row['node'] in G.nodes() and ipv6[row['rpl_parent']] in G.nodes():
            G.add_edge(row['node'], ipv6[row['rpl_parent']])

    return G

