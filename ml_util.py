import os
import pandas as pd
import numpy as np
import pickle
import random
import json
import platform

if platform.system() == 'Linux':
    etc = "/home/apx748/bin/etc"
else:#It is Mac/Darwin
    etc = "/etc"

from math import sqrt, floor, log10

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.font_manager
# from kmodes import kmodes
# from kmodes.kmodes import KModes
from scipy.spatial.distance import cdist, pdist

from numpy import mean

from sklearn.metrics import silhouette_score, r2_score
from sklearn.utils import shuffle
from sklearn.metrics import mean_absolute_error as mae
from sklearn.metrics import mean_squared_error as mse

from sklearn.metrics import auc as roc_auc
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from sklearn.model_selection import cross_val_predict, \
                ParameterGrid, cross_val_score, train_test_split

from paulRegressor import *

HEAD, TAIL = 5, -5

def send_email(subject ="test email", body="testing"):
    import smtplib
    if platform.python_version().split(".")[0] == "2":
        import ConfigParser
        config = ConfigParser.ConfigParser()
        from email.MIMEMultipart import MIMEMultipart
        from email.MIMEText import MIMEText
    else:
        import configparser
        config = configparser.ConfigParser()
        from email.mime.multipart import MIMEMultipart
        from email.mime.text import MIMEText


    sender_config_file = os.path.join(etc, "email_config.txt")
    config.read(sender_config_file)
    SENDER_EMAIL_ID = config.get("configuration","email")

    receiver_config_file = os.path.join(etc, "recipient_config.txt")
    config.read(receiver_config_file)
    RECEIVER_EMAIL_ID = config.get("configuration","email")

    passwd_config_file = os.path.join(etc, "passwd_config.txt")
    config.read(passwd_config_file)
    PASSWD = config.get("configuration","password")

    msg = MIMEMultipart()
    msg['From'] = SENDER_EMAIL_ID
    msg['To'] = RECEIVER_EMAIL_ID
    msg['Subject'] = subject

    msg.attach(MIMEText(body, 'plain'))

    server = smtplib.SMTP('smtp.gmail.com', 587)
    server.starttls()
    server.login(SENDER_EMAIL_ID, PASSWD)
    text = msg.as_string()
    server.sendmail( SENDER_EMAIL_ID, RECEIVER_EMAIL_ID, text)
    server.quit()
    print("Email sent")

def in_jupyter():
    try:
        cfg = get_ipython().config
        return True
    except NameError:
        return False

def round_sig(x, sig=3):
    try:
        return round(x, sig-int(floor(log10(abs(x))))-1)
    except:
        return x

def str_round(x, sig=3):
    return str(round_sig(x,sig))


def mean_absolute_percentage_error(y_true, y_pred):
	y_true = check_arrays(y_true)
	y_pred = check_arrays(y_pred)

	return mean(abs((y_true - y_pred) / y_true)) * 100

def mape(y_true,y_pred):
    return mean_absolute_percentage_error(y_true,y_pred)

def r_score(y_true,y_pred):
    return sqrt(r2_score(y_true,y_pred))

def R_score(y_true,y_pred):
    return round_sig(sqrt(r2_score(y_true,y_pred)))

def R2_score(y_true,y_pred):
    return round_sig(r2_score(y_true,y_pred))

def Mae(y_true,y_pred):
    return round_sig(Mae(y_true,y_pred))

def Mse(y_true,y_pred):
    return round_sig(mse(y_true,y_pred))

def Mape(y_true,y_pred):
    return round_sig(mape(y_true,y_pred))

def crossVal_r2(X,y,estimator,CV=10):
    predicted = cross_val_predict(estimator, X, y, cv=CV)
    return r2_score(y, predicted)

def crossVal_scores(X,y,estimator,CV=10):
    predicted = cross_val_predict(estimator, X, y, cv=CV)
    return r2_score(y, predicted), mape(y,predicted)

def runGrid(algorithm,fpType,label, cv=10, maximum=0.5):
    estimator = getEstimator(algorithm)

    import platform
    if platform.python_version()[0]=='2':
        params, moreParams = loadJson('params'), loadJson('moreParams')
    else:
        params, moreParams = loadData('params'), loadData('moreParams')

    count = 0
    if 'extraTrees' in algorithm or 'randomForest' in algorithm:
        parameters = moreParams[algorithm]
    else:
        parameters = params[algorithm]
    for g in ParameterGrid(parameters):
        count += 1
        estimator.set_params(**g)
        r2,mape = crossVal_scores(fpType,label,estimator,cv)

        if r2>maximum:
            print(estimator)
            print("r2:",r2,"mape:",mape)
            maximum = r2
            best = estimator

    return best

def loadData(name,path='pickles'):
    '''
    This loads a pickle file and returns the content which is a DICTIONARY object in our case.
    '''
    if ".pkl" in name:
            name = name.split(".pkl")[0]
    if "/" in name:
            name = name.split("/",1)[1]

    with open(path+"/"+name + '.pkl', 'rb') as f:
          return pickle.load(f)

def saveData(obj, name,path='pickles'):
    '''
    This saves a object into a pickle file. In our case, it is generally a DICTIONARY object.
    '''

    with open(path+"/"+name + '.pkl', 'wb') as f:
            pickle.dump(obj, f)

def loadNumpy(name,path='data'):
    if ".npy" in name:
        fullPath = path+'/'+name
    else:
        fullPath = path+'/'+name+'.npy'
    return np.load(fullPath)


def saveNumpy(obj, name, path='data'):
    if ".npy" not in name:
        fullPath = path+'/'+name
        np.save(fullPath, obj)
        print(name,'saved successfully in',path)
    else:
        fullPath = path+'/'+name.split(".npy")[0]
        np.save(fullPath, obj)
        print(name,'saved successfully in',path)

def loadJson(name,path='data'):
    if ".json" in name:
        fullPath = path+'/'+name
    else:
        fullPath = path+'/'+name+'.json'
    return json.load(open(fullPath))

def saveJson(obj, name, path='data'):
    if '.json' not in name:
        name = name + '.json'
    fullPath = path+'/'+name
    f = open(fullPath,"w")
    f.write(json.dumps(obj))
    f.close()
    print(name,'saved successfully in',path)

def binTrainTest_80_20_Split(size,save=False):

    N = int(size/5)
    testIndices,trainIndices = [],[]
    index = 0

    for i in range(N):#240 for 1203

        minIndex = (i*5)
        if i ==N-1:#239
            maxIndex = size-1
        else:
            maxIndex = (i*5)+4

        index = random.randint(minIndex,maxIndex)

        for j in range(minIndex,maxIndex+1):

            if j==index:
                testIndices += [j]
            else:
                trainIndices += [j]
    if save:
        saveNumpy(trainIndices,'trainIndices_'+str(size))
        saveNumpy(testIndices,'testIndices_'+str(size))

    return trainIndices, testIndices

def getEstimator(regressor):

	if "lasso" in regressor or "Lasso" in regressor:
		estimator = Lasso(alpha = 0.1)#RandomForestRegressor(random_state=0, n_estimators=100)\
	elif "MultiLasso" in regressor:
		estimator = MultiLasso()
	elif "ridge" in regressor or "Ridge" in regressor:
		estimator = Ridge()#(alphas=[0.1, 1.0, 10.0])
	elif "SGDRegression" in regressor:
		estimator = SGDRegressor()
	elif "NNGarrotteRegression" in regressor:
		estimator = NNGarrotteRegressor()
	elif "KernelRegression" in regressor:
		estimator = KernelRegressor()
	elif "LinearRegression" in regressor:
		estimator = LinearRegression()
	elif "KNeighborsRegression" in regressor:
		estimator = KNeighborsRegressor()
	elif "randomForest" in regressor or "RandomForest" in regressor:
		estimator = RandomForestRegressor()
	elif "extraTrees" in regressor or "ExtraTrees" in regressor:
		estimator = ExtraTreesRegressor()
	elif "rbfSVM" in regressor or "RBFSVM" in regressor:
		estimator = SVR(kernel="rbf")
	elif "linearSVM" in regressor or "LinearSVM" in regressor:
		estimator = SVR(kernel="linear")
	elif "polySVM" in regressor or "PolySVM" in regressor:
		estimator = polySVR()
	elif "ElasticNet" in regressor:
		estimator = ElasticNet()
	elif "MultiElasticNet" in regressor:
		estimator = MultiElasticNet()
	elif "gradientBoost" in regressor or "GradientBoost" in regressor:
		estimator = gradientBoost()
	elif "AdaBoost" in regressor:
		estimator = AdaBoostRegressor()
	elif "Bagging" in regressor:
		estimator = BaggingRegressor()
	elif "DecisionTree" in regressor:
		estimator = DecisionTreeRegressor()
	elif "dummy" in regressor:
		estimator = DummyRegressor()

	return estimator


def cross_val_average(estimator,X,y,n_jobs=-1,cv=10):
    return mean(cross_val_score(estimator=estimator,X=X,y=y,n_jobs=n_jobs,cv=cv))

def min_max_scale(data_1d):
    return np.interp(data_1d, (data_1d.min(), data_1d.max()), (0, 1))

def silhouette(X):
    range_n_clusters = [2, 3, 4, 5, 6,7,8]

    for n_clusters in range_n_clusters:
        # Create a subplot with 1 row and 2 columns
        fig, (ax1, ax2) = plt.subplots(1, 2)
        fig.set_size_inches(18, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1 but in this example all
        # lie within [-0.1, 1]
        ax1.set_xlim([-0.1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        clusterer = KMeans(n_clusters=n_clusters, random_state=10)
        cluster_labels = clusterer.fit_predict(X)

        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(X, cluster_labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(X, cluster_labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[cluster_labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

        # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])

        # 2nd Plot showing the actual clusters formed
        colors = cm.spectral(cluster_labels.astype(float) / n_clusters)
        ax2.scatter(X[:, 0], X[:, 1], marker='.', s=30, lw=0, alpha=0.7,
                    c=colors)

        # Labeling the clusters
        centers = clusterer.cluster_centers_
        # Draw white circles at cluster centers
        ax2.scatter(centers[:, 0], centers[:, 1],
                    marker='o', c="white", alpha=1, s=200)

        for i, c in enumerate(centers):
            ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50)

        ax2.set_title("The visualization of the clustered data.")
        ax2.set_xlabel("Feature space for the 1st feature")
        ax2.set_ylabel("Feature space for the 2nd feature")

        plt.suptitle(("Silhouette analysis for KMeans clustering on sample data "
                      "with n_clusters = %d" % n_clusters),
                     fontsize=14, fontweight='bold')

        plt.show()

def BIC_Elbow_Kmeans(dt_trans,Max=9,Min=2):
    K = range(Min,Max)
    KM = [KMeans(n_clusters=k).fit(dt_trans) for k in K]
    centroids = [k.cluster_centers_ for k in KM]

    D_k = [cdist(dt_trans, cent, 'euclidean') for cent in centroids]
    cIdx = [np.argmin(D,axis=1) for D in D_k]
    dist = [np.min(D,axis=1) for D in D_k]
    avgWithinSS = [sum(d)/dt_trans.shape[0] for d in dist]

    # Total with-in sum of square
    wcss = [sum(d**2) for d in dist]
    tss = sum(pdist(dt_trans)**2)/dt_trans.shape[0]
    bss = tss-wcss

    kIdx = 2

    # elbow curve
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    ax.plot(K[kIdx], avgWithinSS[kIdx], marker='o', markersize=12,
    markeredgewidth=2, markeredgecolor='r', markerfacecolor='None')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Average within-cluster sum of squares')
    plt.title('Elbow for KMeans clustering')

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, bss/tss*100, 'b*-')
    plt.grid(True)
    plt.xlabel('Number of clusters')
    plt.ylabel('Percentage of variance explained')
    plt.title('Elbow for KMeans clustering')

def silhouette_Elbow_Kmeans(dt_trans,Max=9,Min=2):
    s = []
    for n_clusters in range(Min,Max):
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(dt_trans)

        labels = kmeans.labels_
        centroids = kmeans.cluster_centers_

        s.append(silhouette_score(dt_trans, labels, metric='euclidean'))

    x = range(Min,Max)
    plt.plot(x,s)
    plt.ylabel("Silouette")
    plt.xlabel("k")
    plt.title("Silouette for K-means cell's behaviour")
    sns.despine()


def plot_scatter(X, labels=None, centers=None, title="Scatter Plot"):

    labels = np.zeros(shape=X.shape[0], dtype=int) if labels is None else labels
    colors = ['b', 'r', 'g', 'm', 'y']
    col_dict = {}
    i = 0
    for lab in np.unique(labels):
        col_dict[lab] = colors[i]
        i += 1

    fig1 = plt.figure(1, figsize=(8,6))
    ax = fig1.add_subplot(1, 1, 1)

    for i in np.unique(labels):
        indx = np.where(labels == i)[0]
        plt.scatter(X[indx,0], X[indx,1], color=col_dict[i], marker='o', s=100, alpha=0.5)

    if centers is not None:
        plt.scatter(centers[:,0], centers[:,1], color='magenta', marker='*', s=250, alpha=0.5)

    plt.setp(ax.get_xticklabels(), rotation='horizontal', fontsize=16)
    plt.setp(ax.get_yticklabels(), rotation='vertical', fontsize=16)

    plt.xlabel('$x_1$', size=20)
    plt.ylabel('$x_2$', size=20)
    plt.title(title, size=20)

    plt.show()
