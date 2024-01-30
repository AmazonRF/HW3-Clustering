import numpy as np
from scipy.spatial.distance import cdist


class KMeans:
    def __init__(self, k: int, tol: float = 1e-6, max_iter: int = 100):
        """
        In this method you should initialize whatever attributes will be required for the class.

        You can also do some basic error handling.

        What should happen if the user provides the wrong input or wrong type of input for the
        argument k?

        inputs:
            k: int
                the number of centroids to use in cluster fitting
            tol: float
                the minimum error tolerance from previous error during optimization to quit the model fit
            max_iter: int
                the maximum number of iterations before quitting model fit
        """

        """

        Raises:
            ValueError: If k, tol, or max_iter have invalid values.
        """
        # Basic error handling
        # if not isinstance(k, int) or k <= 0:
        #     raise ValueError("k must be a positive integer.")
        # if not isinstance(tol, float) or tol < 0:
        #     raise ValueError("tol must be a positive float.")
        # if not isinstance(max_iter, int) or max_iter <= 0:
        #     raise ValueError("max_iter must be a positive integer.")

        # Initialize attributes
        self.k = k
        self.tol = tol
        self.max_iter = max_iter
        # print("1. max_iter",self.max_iter)

        # self.centers = None  # This will hold the centroids later

    def fit(self, mat: np.ndarray):
        """
        Fits the kmeans algorithm onto a provided 2D matrix.
        As a bit of background, this method should not return anything.
        The intent here is to have this method find the k cluster centers from the data
        with the tolerance, then you will use .predict() to identify the
        clusters that best match some data that is provided.

        In sklearn there is also a fit_predict() method that combines these
        functions, but for now we will have you implement them both separately.

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features
        """

        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input must be a 2D numpy array.")
        
        self.original_data = np.copy(mat)

        # Step 1: Randomly initialize k cluster centers
        randomIniCenter = np.random.choice(mat.shape[0], self.k, replace=False)
        self.centers = mat[randomIniCenter]
        # print("1. randomIniCenter",self.centers)


        for iter in range(self.max_iter):
            # Step 2: Assign each data point to the nearest cluster center
            # assignments = self._assign_points_to_centers(mat)

            # # Step 3: Recalculate the cluster centers
            # new_centers = np.array([mat[assignments == k].mean(axis=0) for k in range(self.k)])

            new_centers = self.get_centroids()
            self.new_centers = new_centers
            # Check for convergence (if centers do not change significantly)
            if np.allclose(self.centers, self.new_centers, atol=self.tol):
                break

            self.centers = new_centers

    def predict(self, mat: np.ndarray) -> np.ndarray:
        """
        Predicts the cluster labels for a provided matrix of data points--
            question: what sorts of data inputs here would prevent the code from running?
            How would you catch these sorts of end-user related errors?
            What if, for example, the matrix is of a different number of features than
            the data that the clusters were fit on?

        inputs:
            mat: np.ndarray
                A 2D matrix where the rows are observations and columns are features

        outputs:
            np.ndarray
                a 1D array with the cluster label for each of the observations in `mat`
        """
       
        if not isinstance(mat, np.ndarray) or mat.ndim != 2:
            raise ValueError("Input must be a 2D numpy array.")

        if self.centers is None:
            raise ValueError("The model is not yet fitted. Please call fit() before predict().")

        if mat.shape[1] != self.centers.shape[1]:
            raise ValueError("The input matrix must have the same number of features as the data used in fit().")

        # Compute the distance from each point to each cluster center
        distances = cdist(mat, self.centers, 'euclidean')

        # Assign each point to the nearest cluster
        cluster_labels = np.argmin(distances, axis=1)

        return cluster_labels

    def get_error(self) -> float:
        """
        Returns the final squared-mean error of the fit model.

        Returns:
            float: The squared-mean error of the fit model.
        """
        
        return np.allclose(self.centers, self.new_centers, atol=self.tol)

    def get_centroids(self):
        # Using scipy's cdist function for distance computation
        new_centers = np.array([self.original_data[self._assign_points_to_centers(self.original_data) == k].mean(axis=0) for k in range(self.k)])

        return new_centers
    
    def _assign_points_to_centers(self, mat):
        distances = np.sqrt(((self.original_data - self.centers[:, np.newaxis])**2).sum(axis=2))
        return np.argmin(distances, axis=0)
    