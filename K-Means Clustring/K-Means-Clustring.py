import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()
from sklearn.cluster import KMeans


def main():

    data=pd.read_csv('3.01. Country clusters.csv')
    plt.scatter(data['Longitude'],data['Latitude'])#plotting the points location
    plt.xlim(-180,180)
    plt.ylim(-90,90)
    plt.show()

    ##### Clustering
    x=data.iloc[:,1:3]#Slicing
    print(x)
    kmeans=KMeans(3) #number of clusters
    kmeans.fit(x)# apply k-means clustring

    ##### Clustring Results
    identifited_clusters=kmeans.fit_predict(x)
    data_with_clusters=data.copy()
    data_with_clusters['Cluster']=identifited_clusters# adding the Cluster col to the data
    print(data_with_clusters)

    ##### plotting the klusters
    plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')  # plotting clusters
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()

if __name__ == '__main__':
    main()