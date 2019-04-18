import pandas as pd
import numpy as np
# KNN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
# SVM
from sklearn import svm
# Import Random Forest Model
from sklearn.ensemble import RandomForestClassifier
import time
# Ensalble
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import LinearSVC




def random_forest_classification(trace_stats, features_to_drop, n_estimators=100, test_size=0.3):
	# INPUT: 
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs 
	######## features_to_drop a list of features to drop
	######## n_estimators number of estimators
	######## test_size the size of test set

	# OUTPUT: return a dataframe containing accuracy, precision, recall and f1-score for each window size
	

	results = None
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]
	    
	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values

	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)
	     
	    #Create a Gaussian Classifier
	    rf_clf = RandomForestClassifier(n_estimators=n_estimators)

	    t0 = time.time()  # Start a timer

	    #Train the model using the training sets y_pred=clf.predict(X_test)
	    rf_clf.fit(X_train,y_train)
	    training_time = time.time() - t0

	    t0 = time.time()  # Start a timer
	    y_pred = rf_clf.predict(X_test)
	    testing_time = time.time() - t0
	    
	    # Add results to a Dataframe
	    if results is None:
	        results = pd.DataFrame({'Model': ['Random Forest'], 
	                                'Window Size': [trace_size], 
	                                'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                'Precision': [metrics.precision_score(y_test, y_pred, average='macro')], 
	                                'Recall': [metrics.recall_score(y_test, y_pred, average='macro')], 
	                                'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                'Training Time (sec)': [training_time],
	                                'Testing Time (sec)': [testing_time]})
	    else:
	        results = pd.concat([results,pd.DataFrame({'Model': ['Random Forest'], 
	                                                         'Window Size': [trace_size], 
	                                                         'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                                         'Precision': [metrics.precision_score(y_test, y_pred, average='macro')], 
	                                                         'Recall': [metrics.recall_score(y_test, y_pred, average='macro')], 
	                                                         'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                                         'Training Time (sec)': [training_time],
	                                                         'Testing Time (sec)': [testing_time]})])


	return results



def random_forest_cross_validation(trace_stats, features_to_drop, n_estimators=100, test_size=0.3, cross_val=5):
	# INPUT: 
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs 
	######## features_to_drop a list of features to drop
	######## n_estimators number of estimators
	######## test_size the size of test set
	######## cross_val the size of cross validation 

	# OUTPUT: return a dataframe containing the mean accuracy

	cv_results = None

	# Select the set of features and labels that we use to fit the algorithm
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]
	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values

	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.3, random_state=1)
	    
	    #Create Random Forest Classifier
	    rf_clf = RandomForestClassifier(n_estimators=100)
	    
	    #train model with cv of 5
	    cv_scores = cross_val_score(rf_clf, features, target, cv = cross_val)
	    
	    if cv_results is None:
	        cv_results = pd.DataFrame({'Model': ['Random Forest'], 
	                                   'Window Size': [trace_size], 
	                                   'Mean Accuracy': [np.mean(cv_scores)]})
	    else:
	        cv_results = pd.concat([cv_results, pd.DataFrame({'Model': ['Random Forest'], 
	                                             'Window Size': [trace_size], 
	                                             'Mean Accuracy': [np.mean(cv_scores)]})])

	return cv_results




def k_nearest_neighbor_classification(trace_stats, features_to_drop, n_estimators=100, test_size=0.3, n_neighbors=3):
	# INPUT: 
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs 
	######## features_to_drop a list of features to drop
	######## n_estimators number of estimators
	######## test_size the size of test set
	######## n_neighbors number of neighbors

	# OUTPUT: return a dataframe containing accuracy, precision, recall and f1-score for each window size

	results = None
	
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]
	    
	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values
	    
	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)
	     
	    #Create KNN Classifier
	    knn = KNeighborsClassifier(n_neighbors=n_neighbors)

	    t0 = time.time()  # Start a timer

	    #Train the model using the training sets
	    knn.fit(X_train, y_train)
	    training_time = time.time() - t0

	    t0 = time.time()	# Start timer
	    #Predict the response for test dataset
	    y_pred = knn.predict(X_test)
	    testing_time = time.time() - t0
	    
	    # Add results to a Dataframe
	    if results is None:
	        results = pd.DataFrame({'Model': ['KNN'], 
	                                'Window Size': [trace_size], 
	                                'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                'Precision': [metrics.precision_score(y_test, y_pred, average='macro')], 
	                                'Recall': [metrics.recall_score(y_test, y_pred, average='macro')], 
	                                'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                'Training Time (sec)': [training_time],
	                                'Testing Time (sec)': [testing_time]})
	    else:
	        results = pd.concat([results,pd.DataFrame({'Model': ['KNN'], 
	                                                         'Window Size': [trace_size], 
	                                                         'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                                         'Precision': [metrics.precision_score(y_test, y_pred, average='macro')], 
	                                                         'Recall': [metrics.recall_score(y_test, y_pred, average='macro')], 
	                                                         'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                                         'Training Time (sec)': [training_time],
	                                                         'Testing Time (sec)': [testing_time]})])


	return results





def k_nearest_neighbor_cross_validation(trace_stats, features_to_drop, n_neighbors=3, test_size=0.3, cross_val=5):
	# INPUT: 
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs 
	######## features_to_drop a list of features to drop
	######## n_estimators number of estimators
	######## test_size the size of test set
	######## cross_val the size of cross validation 

	# OUTPUT: return a dataframe containing the mean accuracy

	cv_results = None

	# Select the set of features and labels that we use to fit the algorithm
	# Select the set of features and labels that we use to fit the algorithm
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]
	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values
	    
	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)
	    
	    #Create KNN Classifier
	    knn_clf = KNeighborsClassifier(n_neighbors=n_neighbors)
	    
	    #train model with cv of 5
	    cv_scores = cross_val_score(knn_clf, features, target, cv = cross_val)
	    
	    if cv_results is None:
	        cv_results = pd.DataFrame({'Model': ['KNN'], 
	                                   'Window Size': [trace_size], 
	                                   'Mean Accuracy': [np.mean(cv_scores)]})
	    else:
	        cv_results = pd.concat([cv_results, pd.DataFrame({'Model': ['KNN'], 
	                                             'Window Size': [trace_size], 
	                                             'Mean Accuracy': [np.mean(cv_scores)]})])

	return cv_results





def support_vector_machines_classification(trace_stats, features_to_drop, kernel='rbf', test_size=0.3):
	# INPUT: 
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs 
	######## features_to_drop a list of features to drop
	######## kernel 
	######## test_size the size of test set

	# OUTPUT: return a dataframe containing accuracy, precision, recall and f1-score for each window size
	
	results = None
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]
	    
	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values
	    
	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)
	     
	    #Create a svm Classifier
	    svm_clf = svm.SVC(kernel=kernel, random_state=9, gamma='scale') # Linear Kernel

	    t0 = time.time()  # Start a timer

	    #Train the model using the training sets
	    svm_clf.fit(X_train, y_train)
	    training_time = time.time() - t0

	    t0 = time.time()	# Start timer
	    #Predict the response for test dataset
	    y_pred = svm_clf.predict(X_test)
	    testing_time = time.time() - t0
	    
	    # Add results to a Dataframe
	    if results is None:
	        results = pd.DataFrame({'Model': ['SVM'], 
	                                'Window Size': [trace_size], 
	                                'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                'Precision': [metrics.precision_score(y_test, y_pred, average='macro')], 
	                                'Recall': [metrics.recall_score(y_test, y_pred, average='macro')], 
	                                'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                'Training Time (sec)': [training_time],
	                                'Testing Time (sec)': [testing_time]})
	    else:
	        results = pd.concat([results,pd.DataFrame({'Model': ['SVM'], 
	                                                         'Window Size': [trace_size], 
	                                                         'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                                         'Precision': [metrics.precision_score(y_test, y_pred, average='macro')], 
	                                                         'Recall': [metrics.recall_score(y_test, y_pred, average='macro')], 
	                                                         'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                                         'Training Time (sec)': [training_time],
	                                                         'Testing Time (sec)': [testing_time]})])



	return results



def support_vector_machines_cross_validation(trace_stats, features_to_drop, kernel='rbf', test_size=0.3, cross_val=5):
	# INPUT: 
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs 
	######## features_to_drop a list of features to drop
	######## n_estimators number of estimators
	######## kernel
	######## cross_val the size of cross validation 

	# OUTPUT: return a dataframe containing the mean accuracy

	cv_results = None

	# Select the set of features and labels that we use to fit the algorithm
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]
	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values
	    
	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)
	    
	    #Create SVM Classifier
	    svm_clf = svm.SVC(kernel=kernel, random_state=9, gamma='scale') # Linear Kernel
	    
	    #train model with cv of 5
	    cv_scores = cross_val_score(svm_clf, features, target, cv = cross_val)
	    
	    if cv_results is None:
	        cv_results = pd.DataFrame({'Model': ['SVM'], 
	                                   'Window Size': [trace_size], 
	                                   'Mean Accuracy': [np.mean(cv_scores)]})
	    else:
	        cv_results = pd.concat([cv_results, pd.DataFrame({'Model': ['SVM'], 
	                                             'Window Size': [trace_size], 
	                                             'Mean Accuracy': [np.mean(cv_scores)]})])

	return cv_results




def ensalble_svm_classification(trace_stats, features_to_drop, n_estimators = 10, test_size=0.3):
	# INPUT:
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs
	######## features_to_drop a list of features to drop
	######## n_estimators number of estimator to use for the ensable
	######## test_size the size of test set

	# OUTPUT: return a dataframe containing accuracy, precision, recall and f1-score for each window size

	results = None
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]

	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values

	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)

	    # Create a SVM Classifier
	    ovr_clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(random_state=9, max_iter=10000, C=1.2), max_samples=1.0/n_estimators,
	                                                    n_estimators=n_estimators))


	    t0 = time.time()	# Start timer
	    #Train the model using the training sets
	    ovr_clf.fit(X_train, y_train)
	    training_time = time.time() - t0

	    t0 = time.time()	# Start timer
	    #Predict the response for test dataset
	    y_pred = ovr_clf.predict(X_test)
	    testing_time = time.time() - t0


	    # Add results to a Dataframe
	    if results is None:
	        results = pd.DataFrame({'Model': ['OneVsRestClassifier (SVM)'],
	                                'Window Size': [trace_size],
	                                'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                'Precision': [metrics.precision_score(y_test, y_pred, average='macro')],
	                                'Recall': [metrics.recall_score(y_test, y_pred, average='macro')],
	                                'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                'Training Time (sec)': [training_time],
	                                'Testing Time (sec)': [testing_time]})
	    else:
	        results = pd.concat([results,pd.DataFrame({'Model': ['OneVsRestClassifier (SVM)'],
	                                                         'Window Size': [trace_size],
	                                                         'Accuracy': [metrics.accuracy_score(y_test, y_pred)],
	                                                         'Precision': [metrics.precision_score(y_test, y_pred, average='macro')],
	                                                         'Recall': [metrics.recall_score(y_test, y_pred, average='macro')],
	                                                         'F1-score': [metrics.f1_score(y_test, y_pred, average='macro')],
	                                                         'Training Time (sec)': [training_time],
	                                                         'Testing Time (sec)': [testing_time]})])



	return results






def ensalble_svm_cross_validation(trace_stats, features_to_drop, n_estimators = 10, test_size=0.3, cross_val=5):
	# INPUT:
	######## trace_stats a dictionary containing (window_size, statistics per node) pairs
	######## features_to_drop a list of features to drop
	######## n_estimators number of estimators
	######## test_size the size of the test set
	######## cross_val the size of cross validation

	# OUTPUT: return a dataframe containing the mean accuracy

	cv_results = None

	# Select the set of features and labels that we use to fit the algorithm
	for trace_size in trace_stats:
	    print('Computing trace {}'.format(trace_size))
	    trace = trace_stats[trace_size]
	    # separate features from target values
	    features = trace.drop(columns=features_to_drop)
	    target = trace['label'].values

	    # split dataset into train and test data
	    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=test_size, random_state=1)

	    # create SVM Classifier
	    ovr_clf = OneVsRestClassifier(BaggingClassifier(LinearSVC(random_state=9, max_iter=10000, C=1.2), max_samples=1.0/n_estimators,
	                                                    n_estimators=n_estimators))

	    # train model with cv of 5
	    cv_scores = cross_val_score(ovr_clf, features, target, cv = cross_val)

	    if cv_results is None:
	        cv_results = pd.DataFrame({'Model': ['OneVsRestClassifier (SVM)'],
	                                   'Window Size': [trace_size],
	                                   'Mean Accuracy': [np.mean(cv_scores)]})
	    else:
	        cv_results = pd.concat([cv_results, pd.DataFrame({'Model': ['OneVsRestClassifier (SVM)'],
	                                             'Window Size': [trace_size],
	                                             'Mean Accuracy': [np.mean(cv_scores)]})])

	return cv_results

