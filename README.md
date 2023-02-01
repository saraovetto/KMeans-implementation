# K-Means algorithm

I developed this Python script as a project for the Scientific Programming course delivered by the Politecnico of Milan as a part of the MSc in Bioinformatics for Computational Genomics.

This implementation provides two different versions of the K-means clustering algorithm: one based on lists and one based on numpy arrays.

#### Description
The K means clustering algorithm starts by initializating k centroids chosen randomly from the dataset, then applies iteratively two steps: in the **assignment** the points are assigned to the closest centroid in terms of euclidean distance and the clusters are formed, then the position of the centroids is **updated** as the mean point of each cluster. The algorithm proceeds until convergence, so when the proper clustering solution is reached.

The convergence criteria used in this implementation is the almost-equality of the positions of old centroids and new centroids: the algorithm stops when the last n complete iterations (default is n=5) do not move any of the centroids by a distance bigger than the defined tolerance value.

The two different versions are compared in terms of execution time and their performance is tested when changes in vector size (point dimensions), number of vectors (sample size) and number of clusters occurr. 
