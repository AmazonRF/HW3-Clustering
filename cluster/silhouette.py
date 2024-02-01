import numpy as np
from scipy.spatial.distance import cdist


class Silhouette:
    def __init__(self):
        """
        inputs:
            none
        """
        # self.outscore: mean Silhouette score
        # self.perpointscore: Silhouette score


    def score(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        calculates the silhouette score for each of the observations

        inputs:
            X: np.ndarray
                A 2D matrix where the rows are observations and columns are features.

            y: np.ndarray
                a 1D array representing the cluster labels for each of the observations in `X`

        outputs:
            np.ndarray
                a 1D array with the silhouette scores for each of the observations in `X`
        """

        k = len(set(y))
        
        if k == 1:
            raise ValueError("Silhouette score cannot be calculated for a single cluster.")
        
        # Calculate pairwise distances between points
        all_distances = cdist(X, X)
        
        # Calculate a and b for each point
        a = np.array([np.mean(all_distances[i, y == y[i]]) for i in range(X.shape[0])])

        # b = np.array([np.min([np.mean(all_distances[i, labels == label]) for label in range(k) if label != labels[i]]) for i in range(X.shape[0])])
    
         # Calculate silhouette score for each point
        b = []
        for i in range(X.shape[0]):
            b_value = []
            for label in range(k):
                if label != y[i]:
                    b_value.append([np.mean(all_distances[i, y == label])])
            b.append(np.min(b_value))

        b = np.array(b)
        
        # Calculate silhouette score for each point
        s = (b - a) / np.maximum(a, b)

        #output
        self.outscore = np.mean(s)
        self.perpointscore = s
        return s
