# ----------------------------------------------------------------
# IoT Netprofiler
# Licensed under The MIT License [see LICENSE for details]
# Written by Luca Maiano - https://www.linkedin.com/in/lucamaiano/
# ----------------------------------------------------------------



import pandas as pd
import numpy as np
import os
from os import listdir

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import auc

from sklearn.preprocessing import LabelBinarizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
import time

import tensorflow as tf
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.optimizers import SGD
from keras.wrappers.scikit_learn import KerasClassifier
from keras import regularizers

from lib.visualization import data_visualization


# -----------------------------------------------
# 			Supervised Learning
# -----------------------------------------------


def random_forest_classifier(X_train, y_train, X_test, y_test, n_estimators=100, cross_val=5):
	# INPUT: 
	######## X_train, y_train, X_test, y_test: features and labels of train set and features and labels of test set respectively
	######## n_estimators number of estimators
	######## cross_val the size of cross validation 

	# OUTPUT: return a set of prediction

	clf = RandomForestClassifier(n_estimators=n_estimators)
	
	cv = StratifiedKFold(n_splits=cross_val, random_state=123, shuffle=True)
	cross = 1
	auc_scores = []
	    
	for (train, test), i in zip(cv.split(X_train, y_train), range(cross_val)):
	    clf.fit(X_train.iloc[train], y_train.iloc[train])

	    y = y_train.iloc[test]
	    pred = clf.predict(X_train.iloc[test])
	    if len(set(y_train)) <= 2:
		    fpr, tpr, thresholds = roc_curve(y, pred)
		    auc_score = auc(fpr, tpr)
	    else:
		    auc_score = multiclass_roc_auc_score(y, pred)

	    print('AUC on validation set {}/{}: {}'.format(cross, cross_val, auc_score))
	    cross += 1

	auc_scores.append(auc_score)
	print("Mean AUC %.3f (Std +/- %.3f)" % (np.mean(auc_scores), np.std(auc_scores)))
	
	y_pred = clf.predict(X_test)

	return y_pred



def k_nn_classifier(X_train, y_train, X_test, y_test, n_neighbors=3, cross_val=5):
	# INPUT: 
	######## X_train, y_train, X_test, y_test: features and labels of train set and features and labels of test set respectively
	######## n_neighbors number of neighbors
	######## cross_val size of cross validation

	# OUTPUT: return a set of prediction

	clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	
	cv = StratifiedKFold(n_splits=cross_val, random_state=123, shuffle=True)
	cross = 1
	auc_scores = []
	    
	for (train, test), i in zip(cv.split(X_train, y_train), range(cross_val)):
	    clf.fit(X_train.iloc[train], y_train.iloc[train])

	    y = y_train.iloc[test]
	    pred = clf.predict(X_train.iloc[test])
	    if len(set(y_train)) <= 2:
		    fpr, tpr, thresholds = roc_curve(y, pred)
		    auc_score = auc(fpr, tpr)
	    else:
		    auc_score = multiclass_roc_auc_score(y, pred)

	    auc_scores.append(auc_score)
	    print('AUC on validation set {}/{}: {}'.format(cross, cross_val, auc_score))
	    cross += 1

	print("Mean AUC %.3f (Std +/- %.3f)" % (np.mean(auc_scores), np.std(auc_scores)))

	y_pred = clf.predict(X_test)

	return y_pred


def svm_classifier(X_train, y_train, X_test, y_test, kernel='rbf', cross_val=5):
	# INPUT: 
	######## X_train, y_train, X_test, y_test: features and labels of train set and features and labels of test set respectively
	######## kernel
	######## cross_val size of cross validation

	# OUTPUT: return a set of prediction

	clf = svm.SVC(kernel=kernel, random_state=13, gamma='auto', decision_function_shape='ovr')
	
	cv = StratifiedKFold(n_splits=cross_val, random_state=123, shuffle=True)
	cross = 1
	auc_scores = []
	    
	for (train, test), i in zip(cv.split(X_train, y_train), range(cross_val)):
	    clf.fit(X_train.iloc[train], y_train.iloc[train])

	    y = y_train.iloc[test]
	    pred = clf.predict(X_train.iloc[test])
	    if len(set(y_train)) <= 2:
		    fpr, tpr, thresholds = roc_curve(y, pred)
		    auc_score = auc(fpr, tpr)
	    else:
		    auc_score = multiclass_roc_auc_score(y, pred)

	    auc_scores.append(auc_score)
	    print('AUC on validation set {}/{}: {}'.format(cross, cross_val, auc_score))
	    cross += 1

	print("Mean AUC %.3f (Std +/- %.3f)" % (np.mean(auc_scores), np.std(auc_scores)))

	y_pred = clf.predict(X_test)

	return y_pred




def neural_net_classifier(X_train, y_train, X_test, y_test, model_name, epochs=1500, batch_size=64, corss_val=5):
    def create_model():
        # define the keras model
        model = Sequential()
        model.add(Dense(32, input_dim=len(X_train.columns), activation='relu'))
        model.add(Dropout(0.2))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)))
        model.add(Dense(96, activation='relu', kernel_regularizer=regularizers.l1(0.003)))
        model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l1(0.001)))
        model.add(Dense(32, activation='relu', kernel_regularizer=regularizers.l1(0.001)))
        model.add(Dropout(0.3))
        if len(set(y_train)) <= 2:
        	model.add(Dense(1, activation='sigmoid'))
        else:
        	model.add(Dense(len(set(y_train)), activation='sigmoid'))

        # Compile model
        if len(set(y_train)) <= 2:
            model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
        else:
            model.compile(loss='sparse_categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])

        #model.fit(trainX, trainy, epochs=300, verbose=0)

        return model


    # Cross Validation
    k_clf = KerasClassifier(build_fn=create_model, epochs=epochs, batch_size=batch_size, verbose=0)
    kfold = StratifiedKFold(n_splits=corss_val, shuffle=True, random_state=15)
    results = cross_val_score(k_clf, X_train, y_train, cv=kfold, scoring='accuracy')
    print("Mean Accuracy: %.2f%% (%.2f%%)" % (results.mean()*100, results.std()*100))

    clf = create_model()
    history = clf.fit(X_train, y_train, validation_split=0.2, epochs=epochs, batch_size=batch_size, shuffle=True, verbose=0)

    y_pred = clf.predict_classes(X_test, verbose=0)

    data_visualization.visualize_loss_function(model_name, history)
    if 'models' not in listdir(os.getcwd()+'/data/'):
            os.makedirs('data/models')
    clf.save("data/models/" + model_name + "_" + "model.h5")

    try:
    	y_pred = y_pred[:, 0]
    	
    	return y_pred
    except:
    	return y_pred



# -----------------------------------------------
# 			Unsupervised Learning
# -----------------------------------------------

def pca_transformation(X, n_components=3):
	# INPUT: 
	######## set of data points X
	######## number of components n_components

	# OUTPUT: return transformed set of points

	pca = PCA(n_components=n_components)
	principal_components = pca.fit_transform(X)
	X_pca = pd.DataFrame(data=principal_components)

	return X_pca



def kmeans_classifier(X, n_clusters=2):
	# INPUT: 
	######## X_train, y_train, X_test, y_test: features and labels of train set and features and labels of test set respectively
	######## kernel
	######## cross_val size of cross validation

	# OUTPUT: return a set of prediction

	# Number of clusters
	clf = KMeans(n_clusters=n_clusters, init='k-means++', n_init=20, n_jobs=4, random_state=7)
	# Fitting the input data
	clf.fit(X)
	# Getting the cluster labels
	y_pred = clf.predict(X)
	# Centroid values
	centroids = clf.cluster_centers_

	return y_pred, centroids



# -----------------------------------------------
# 				Metrics
# -----------------------------------------------


def test_metrics(model, y_test, y_pred):
    # accuracy: (tp + tn) / (p + n)
    accuracy = accuracy_score(y_test, y_pred)

    if len(y_test) <= 2:
	    # precision tp / (tp + fp)
	    precision = precision_score(y_test, y_pred)
	    # recall: tp / (tp + fn)
	    recall = recall_score(y_test, y_pred)
	    # f1: 2 tp / (2 tp + fp + fn)
	    f1 = f1_score(y_test, y_pred)
	    # ROC AUC
	    auc = roc_auc_score(y_test, y_pred)
    else:
		# precision tp / (tp + fp)
	    precision = multiclass_precision(y_test, y_pred)
	    # recall: tp / (tp + fn)
	    recall = multiclass_recall(y_test, y_pred)
	    # f1: 2 tp / (2 tp + fp + fn)
	    f1 = multiclass_f1(y_test, y_pred)
	    # ROC AUC
	    auc = multiclass_roc_auc_score(y_test, y_pred)
    
    results = pd.DataFrame({'model': [model], 
                            'accuracy': [accuracy],
                            'precision': [precision],
                            'recall': [recall],
                            'f1-score': [f1],
                            'auc roc': [auc]})
    
    # confusion matrix
    matrix = confusion_matrix(y_test, y_pred)
    
    return results, matrix



def multiclass_precision(y_test, y_pred, average="macro"):
	# Code By: Eric Plog
	# INPUT: 
	######## y_test, y_pred
	######## average

	# OUTPUT: return a roc auc score for multple classes

	lb = LabelBinarizer()
	lb.fit(y_test)
	y_test = lb.transform(y_test)
	y_pred = lb.transform(y_pred)

	return precision_score(y_test, y_pred, average=average)



def multiclass_recall(y_test, y_pred, average="macro"):
	# Code By: Eric Plog
	# INPUT: 
	######## y_test, y_pred
	######## average

	# OUTPUT: return a roc auc score for multple classes

	lb = LabelBinarizer()
	lb.fit(y_test)
	y_test = lb.transform(y_test)
	y_pred = lb.transform(y_pred)

	return recall_score(y_test, y_pred, average=average)


def multiclass_f1(y_test, y_pred, average="macro"):
	# Code By: Eric Plog
	# INPUT: 
	######## y_test, y_pred
	######## average

	# OUTPUT: return a roc auc score for multple classes

	lb = LabelBinarizer()
	lb.fit(y_test)
	y_test = lb.transform(y_test)
	y_pred = lb.transform(y_pred)

	return f1_score(y_test, y_pred, average=average)


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
	# Code By: Eric Plog
	# INPUT: 
	######## y_test, y_pred
	######## average

	# OUTPUT: return a roc auc score for multple classes

	lb = LabelBinarizer()
	lb.fit(y_test)
	y_test = lb.transform(y_test)
	y_pred = lb.transform(y_pred)

	return roc_auc_score(y_test, y_pred, average=average)



# -----------------------------------------------
# 				Results
# -----------------------------------------------
def write_results(models_results, feature_list, n_classes=2,):
	# INPUT: 
	######## list of model results
	######## feature_list: list of features used during training 
	######## number of labels used during training n_classes

	# OUTPUT: return a roc auc score for multple classes
    
    columns = ['model','n_classes','accuracy','precision','recall','f1-score','auc roc','features']
    final_results= pd.DataFrame(columns=columns) 
    features = ''
    for feat in sorted(feature_list):
    	if len(features) == 0:
    		features += feat
    	else:
    		features += str(', ') + feat
    for model in models_results:
        model = model.join(pd.DataFrame({'n_classes': [n_classes], 'features': [features]}))
        final_results = pd.concat([final_results, model])
        
    final_results = final_results.reset_index(drop=True)
    if 'results' not in listdir(os.getcwd() + '/data/'):
            # If destination folders do not exist
            os.makedirs('data/results')
    final_results.to_csv('data/results/'+str(n_classes)+'classes_results.csv')
    
    return final_results[columns]