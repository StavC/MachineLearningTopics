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
    kmeans.fit(x)# apply k-means clustring on x

    ##### Clustring Results
    identifited_clusters=kmeans.fit_predict(x)
    data_with_clusters=data.copy()
    data_with_clusters['Cluster']=identifited_clusters# adding the Cluster col to the data
    print(data_with_clusters)

    ##### plotting the clusters
    plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'],c=data_with_clusters['Cluster'],cmap='rainbow')  # plotting clusters
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()



    ####################################################################
    ####################################################################
    #clustring by language
    data_by_language=data.copy()
    data_by_language['Language']=data_by_language['Language'].map({'English':0, 'French':1,'German':2})
    x = data_by_language.iloc[:, 3:4]  # Slicing
    print(x)
    ##### Clustring
    kmeans = KMeans(3)  # number of clusters
    kmeans.fit(x)  # apply k-means clustring on x

    ##### Clustring Results
    identifited_clusters = kmeans.fit_predict(x)
    data_with_clusters = data_by_language.copy()
    data_with_clusters['Cluster'] = identifited_clusters  # adding the Cluster col to the data
    print(data_with_clusters)

    #plotting
    plt.scatter(data_with_clusters['Longitude'], data_with_clusters['Latitude'], c=data_with_clusters['Cluster'],
                cmap='rainbow')  # plotting clusters
    plt.xlim(-180, 180)
    plt.ylim(-90, 90)
    plt.show()

    ####################################################################
    ####################################################################
    #WCSS and Elbow Method to decied how many clusters we shoudld have
    wcss=[]
    for i in range(1,7):
        kmeans=KMeans(i)
        kmeans.fit(x)
        wcss_iter=kmeans.inertia_
        wcss.append(wcss_iter)

    print(wcss)

    ## ploting the elbow
    number_clusters=range(1,7)
    plt.plot(number_clusters,wcss)
    plt.title("The Elbow  Method")
    plt.xlabel("number of clusters")
    plt.ylabel("Within-Cluster Sum of Squares")
    plt.show()

if __name__ == '__main__':
    main()