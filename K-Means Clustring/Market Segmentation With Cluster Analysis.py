import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans
from sklearn import preprocessing



def main():

    data =pd.read_csv('3.12. Example.csv')
    print(data)

    ##### plot the data
    plt.scatter(data['Satisfaction'],data['Loyalty'])
    plt.xlabel('Statisfaction')
    plt.ylabel('Loyalty')
    plt.show()### seems like two clusters

    ##### Select the Features
    x=data.copy()# both features

    ##### Clustering
    kmeans=KMeans(2)
    kmeans.fit(x)

    ##### Clustering Results

    clusters=x.copy()
    clusters['cluster_pred']=kmeans.fit_predict(x)
    print(clusters)
    plt.scatter(clusters['Satisfaction'],clusters['Loyalty'],c=clusters['cluster_pred'],cmap='rainbow')
    plt.xlabel('Statisfaction')
    plt.ylabel('Loyalty')
    plt.show() ## we can see there is higher weight to Statisfaction compare to Loyalty becasue Loyalty is standart and much smaller
    # we need to give them equal weight-Standardize Statsfication

    ##### Standardize the Variables
    x_scaled=preprocessing.scale(x)
    print(x_scaled)


    ##### Take advantage of the Elbow method

    wcss=[]
    for i in range(1,10):
        kmeans=KMeans(i)
        kmeans.fit(x_scaled)
        wcss.append(kmeans.inertia_)

    plt.plot(range(1,10), wcss)
    plt.title("The Elbow  Method")
    plt.xlabel("number of clusters")
    plt.ylabel("Within-Cluster Sum of Squares")
    plt.show()

    ##### Explore clustering solutions and select the number of clusters

    kmeans_new=KMeans(3)# we should try 2,3,4,5 becasue of the elbow
    kmeans_new.fit(x_scaled)
    clusters_new=x.copy()
    clusters_new['cluster_pred']=kmeans_new.fit_predict(x_scaled)


    ##### plotting the cluster
    plt.scatter(clusters_new['Satisfaction'], clusters_new['Loyalty'], c=clusters_new['cluster_pred'], cmap='rainbow')
    plt.xlabel('Statisfaction')
    plt.ylabel('Loyalty')
    plt.show()

if __name__ == '__main__':
    main()