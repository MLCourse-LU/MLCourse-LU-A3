import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.ticker import MaxNLocator

from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_samples, silhouette_score, completeness_score, homogeneity_score
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids

if __name__ == '__main__':

    """Task 1
    Implement code that does the following:
    - Apply KMeans and KMedoids using k=5
    - Make three plots side by side, showing the clusters identified by the models and the ground truth
    - Plots should have titles, axis labels, and a legend.
    - The plots should also show the centroids of the KMeans and KMedoids clusters.
      Use a different marker style to make them clearly identifiably.
    """

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/vehicles.csv'

    x = df['weight']
    y = df['speed']
    c = df['label']
    k=5 
    model = KMeans(n_clusters=k)
    model.fit(X)
    cx = model.cluster_centers_[:,0]
    cy = model.cluster_centers_[:,1]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)
    ax1.scatter(x=x, y=y,c = model.labels_, s=5, label='Data Points')
    ax1.scatter(x = cx, y=cy,c = 'r', s=50, marker = 'x', label='Centroids') 
    ax1.legend(["Data Points", "Centroids"], loc="lower right")
    ax1.set_title('KMeans')
    ax1.set_xlabel('weight')
    ax1.set_ylabel('speed')

    model = KMedoids(n_clusters=k)
    model.fit(X)
    cx = model.cluster_centers_[:,0] 
    cy = model.cluster_centers_[:,1]
    ax2.scatter(x=x, y=y, c = model.labels_, s=5, label='Data Points')
    ax2.scatter(x=cx, y=cy, c = 'r', marker = 'x', s=50, label='Centroids') 
    ax2.legend(["Data Points", "Centroids"], loc="lower right")
    ax2.set_title('KMedoids')
    ax2.set_xlabel('weight')
    scatter = ax3.scatter(x=x, y=y, c=c, s=5) 
    legend1 = ax3.legend(*scatter.legend_elements(), loc="lower right", title="Classes")
    ax3.add_artist(legend1)
    ax3.set_title('Ground truth')
    ax3.set_xlabel('weight')
    plt.show()

    plt.tight_layout()
    plt.savefig('Figure1.pdf')  # save as PDF to get the nicest resolution in your report.
    plt.show()

    """ Task 2
    Apply KMeans and KMedoids to the following dataset. The choice of K is up to you.
    - Make plots of the best results you got with KMeans and KMedoids.
    - In the title of the plots, indicate the K used, and the homogeneity and completeness score achieved.
    """

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-2.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)
    X = df.iloc[:, :-1].values  # all except the last column
    y = df.iloc[:, -1].values  # the last column
    feature_names = df.columns[:-1]
    k = 4

    url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-2.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)

    X = df.iloc[:, :-1].values  # all except the last column
    y = df.iloc[:, -1].values  # the last column

    k = 4

    model = KMeans(n_clusters=k)
    model.fit(X) 
    y_pred = model.predict(X)

    hom1_sc = homogeneity_score(y, y_pred)
    com1_sc = completeness_score(y, y_pred)

    x = df['X0']
    y = df['X1']
    c = df['Y']

    model = KMeans(n_clusters=k)
    model.fit(X)
    cx = model.cluster_centers_[:,0]
    cy = model.cluster_centers_[:,1]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,5), sharex=True, sharey=True)
    ax1.scatter(x=x, y=y,c = model.labels_, s=5, label='True labels')
    ax1.scatter(x = cx, y=cy,c = 'r', s=50, marker = 'x', label='Centroids') 
    ax1.legend(["True labels", "Centroids"], loc="lower left",title="Data")
    ax1.set_title('KMeans with k=4 \n homogeneity score is ' +str(hom1_sc)+ '\n completeness score is ' +str(com1_sc))
    ax1.set_xlabel('X0')
    ax1.set_ylabel('X1')

    X = df.iloc[:, :-1].values  # all except the last column
    y = df.iloc[:, -1].values  # the last column

    k = 4
    
    model = KMeans(n_clusters=k)
    model.fit(X) 
    y_pred = model.predict(X)
    hom2_sc = homogeneity_score(y, y_pred)
    com2_sc = completeness_score(y, y_pred)

    x = df['X0']
    y = df['X1']
    c = df['Y']

    model = KMedoids(n_clusters=k)
    model.fit(X)
    cx = model.cluster_centers_[:,0] 
    cy = model.cluster_centers_[:,1]
    ax2.scatter(x=x, y=y, c = model.labels_, s=5, label='True labels')
    ax2.scatter(x=cx, y=cy, c = 'r', marker = 'x', s=50, label='Centroids')
    ax2.legend(["True labels", "Centroids"], loc="lower left",title="Data")
    ax2.set_title('KMedoids with k=4 \n homogeneity score is ' +str(hom2_sc)+ '\n completeness score is ' +str(com2_sc))
    ax2.set_xlabel('X0')
    scatter = ax3.scatter(x=x, y=y, c=c, s=5) 
    legend1 = ax3.legend(*scatter.legend_elements(), loc="lower left", title="Classes")
    ax3.add_artist(legend1)
    ax3.set_title('Ground truth')
    ax3.set_xlabel('X0')
    plt.show()

    plt.savefig('Figure2.pdf')
    plt.show()

    """ Task 3

    Adapt the code used in the example to instead make a comparison between KMeans and KMedoids.
    - Set K at 4
    - Make a plot for both models
    """
    url = f'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-3.csv'
    df = pd.read_csv(filepath_or_buffer=url, header=0)
    X = df.values  # convert from pandas to numpy
    n_clusters = 4

    #plots for KMeans
    clusterer = KMeans(n_clusters=4, random_state=10) 
    cluster_labels = clusterer.fit_predict(X)

    n_clusters = 4

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is :{silhouette_avg}")
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_title("The silhouette plot for the various clusters.")

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_xlim([-0.1, 1])
    ax1.set_xticks([-0.6, -0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    ax1.set_ylabel("Cluster label")
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    ax1.set_yticks([])  # Clear the yaxis labels / ticks

    
    y_lower = 10 
    
    for i in range(4): 
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        y_range = np.arange(y_lower, y_upper)
    
        color = cm.nipy_spectral(float(i) / n_clusters)

        ax1.fill_betweenx(y=y_range,                           
                        x1=0,                                 
                        x2=ith_cluster_silhouette_values,   
                        facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10   

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--") 

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(f"Silhouette analysis for KMeans clustering on sample data with n_clusters = {n_clusters}",
                    fontsize=14, fontweight='bold')
 

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters) 
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7, c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')   

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    plt.show()
    
    #plots for KMedoids
    clusterer = KMedoids(n_clusters=4, random_state=10) 
    cluster_labels = clusterer.fit_predict(X)

    n_clusters = 4

    silhouette_avg = silhouette_score(X, cluster_labels)
    print(f"For n_clusters = {n_clusters}, the average silhouette_score is :{silhouette_avg}")
      
    sample_silhouette_values = silhouette_samples(X, cluster_labels)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(18, 7)

    ax1.set_title("The silhouette plot for the various clusters.")

    ax1.set_xlabel("The silhouette coefficient values")
    ax1.set_xlim([-0.1, 1])
    ax1.set_xticks([-0.6, -0.4,-0.2, 0, 0.2, 0.4, 0.6, 0.8, 1])

    ax1.set_ylabel("Cluster label")
    ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
    ax1.set_yticks([])  # Clear the yaxis labels / ticks
    
    y_lower = 10 

    for i in range(4): 
        ith_cluster_silhouette_values = sample_silhouette_values[cluster_labels == i]
        ith_cluster_silhouette_values.sort()

        size_cluster_i = ith_cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        y_range = np.arange(y_lower, y_upper)
    
        color = cm.nipy_spectral(float(i) / n_clusters)

        ax1.fill_betweenx(y=y_range,                        
                        x1=0,                                
                        x2=ith_cluster_silhouette_values,    
                        facecolor=color, edgecolor=color, alpha=0.7)
        ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

        y_lower = y_upper + 10 

    ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

    ax2.set_title("The visualization of the clustered data.")
    ax2.set_xlabel("Feature space for the 1st feature")
    ax2.set_ylabel("Feature space for the 2nd feature")

    plt.suptitle(f"Silhouette analysis for KMedoids clustering on sample data with n_clusters = {n_clusters}",
                    fontsize=14, fontweight='bold')

    colors = cm.nipy_spectral(cluster_labels.astype(float) / n_clusters) 
    ax2.scatter(X[:, 0], X[:, 1], marker='.', s=100, lw=0, alpha=0.7, c=colors, edgecolor='k')

    centers = clusterer.cluster_centers_
    ax2.scatter(centers[:, 0], centers[:, 1], marker='o', c="white", alpha=1, s=200, edgecolor='k')
 

    for i, c in enumerate(centers):
        ax2.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')
    
    plt.show()    

    """ FINISH """
    plt.savefig('Figure3.pdf')
    plt.show()  # show all the plots, in the order they were generated

    """Task 4

    Write code to generate elbow plots for the datasets given here.
    Use them to figure out the likely K for each of them.
    Put the plots you used to make your decision in your report.
    """
    for dataset in ['4a', '4b']:
        url = f'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-{dataset}.csv'
        df = pd.read_csv(filepath_or_buffer=url, header=0)
        X = df.values

        #elbow method for 4a
        url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-4a.csv'
        df = pd.read_csv(filepath_or_buffer=url, header=0)

        X = df.loc[:,['X0', 'X1', 'X2']].values

        inertia = []
        for k in range(1, 9):
            k_means = KMeans(n_clusters=k, init='random')
            k_means.fit(X)
            inertia.append(k_means.inertia_)
    
        plt.figure(figsize=(15,8))

        plt.plot(range(1, 9), inertia, linewidth=2, marker='o')
        plt.title('Elbow method KMeans dataset 4a\n', fontsize=18)
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.show()
        #external source: 'https://neptune.ai/blog/clustering-algorithms'
        
        #elbow method for 4b
        url = 'https://raw.githubusercontent.com/MLCourse-LU/Datasets/main/dataset-task-4b.csv'
        df = pd.read_csv(filepath_or_buffer=url, header=0)

        X = df.loc[:,['X0', 'X1', 'X2', 'X3', 'X4']].values

        inertia = []
        for k in range(1, 9):
            k_means = KMeans(n_clusters=k, init='random')
            k_means.fit(X)
            inertia.append(k_means.inertia_)
    
        plt.figure(figsize=(15,8))

        plt.plot(range(1, 9), inertia, linewidth=2, marker='o')
        plt.title('Elbow method KMeans dataset 4b\n', fontsize=18)
        plt.xlabel('k')
        plt.ylabel('inertia')
        plt.show()
        #external source: 'https://neptune.ai/blog/clustering-algorithms'

        plt.tight_layout()
        plt.savefig(f'Figure 4-{dataset}.pdf')
        plt.show()

    """ Task 5

    Write code that generates a dataset with k >= 3 and 2 feature dimensions.
    - It should be easy for a human to cluster with the naked eye.
    - It should NOT be easy for KMedoids to cluster, even when using the correct value of K.
    - Plot the ground truth of your dataset, so that we can see that a human indeed clusters it easily.
    - Plot the clustering found by KMedoids to show that it doesn't do it well.
    """

    # initialize data of lists.
    d = {'X0':[0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4],
        'X1':[1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9,1,2,3,4,5,6,7,8,9],
        'Y': [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,4,4,4,4,4,4,4,4,4]}
 
    # Create DataFrame
    df1 = pd.DataFrame(d)

    x = df1['X0']
    y = df1['X1']  
    c = df1['Y']

    X = df1.iloc[:, :-1] 
    
    k=5 
    model = KMedoids(n_clusters=k) 
    model.fit(X)
    cx = model.cluster_centers_[:,0]
    cy = model.cluster_centers_[:,1]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15,5), sharex=True, sharey=True)
    ax1.scatter(x=x, y=y, c=model.labels_, s=25, label='Data Points')
    ax1.scatter(x = cx, y=cy, c = 'r', s=75, marker = "x", label='Centroids') 
    ax1.legend(["Data Points", "Centroids"],loc="lower left")
    ax1.set_title('KMedoids')
    ax1.set_xlabel('X0')
    ax1.set_ylabel('X1')

    ax2.scatter(x=x, y=y, c=c, s=25)
    legend1 = ax2.legend(*scatter.legend_elements(),
                        loc="lower left", title="Classes")
    ax2.add_artist(legend1)
    ax2.set_title('Ground truth')
    ax2.set_xlabel('X0')
    plt.show()

    plt.tight_layout()
    plt.savefig('Figure5.pdf')
    plt.show()

