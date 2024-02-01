
import numpy as np
import pytest
from cluster import silhouette  
from cluster import KMeans 
from sklearn.metrics import silhouette_score
from cluster.utils import (
        make_clusters, 
        plot_clusters,
        plot_multipanel)

# write your silhouette score unit tests here

def test_silhouette_scores_known_dataset():
    # A small dataset with known cluster assignments
    X = np.array([[1, 2], [1, 4], [1, 0],
                  [10, 2], [10, 4], [10, 0]])
    labels = np.array([0, 0, 0, 1, 1, 1])

    expected_scores = np.mean([0.85422257, 0.7862365,  0.7862365,  0.85422257 ,0.7862365 , 0.7862365 ]) 
    calculated_scores = silhouette.Silhouette()
    # Calculate silhouette scores
    calculated_scores.score(X = X, y = labels)
    np.testing.assert_allclose(calculated_scores.outscore, expected_scores, atol=0.1)

def test_invalid_k():
    X = np.array([[1, 2], [1, 4], [1, 0]])
    labels = np.array([0, 0, 0])
    
    # Check if ValueError is raised for single cluster
    with pytest.raises(ValueError):
        calculated_scores = silhouette.Silhouette()
        # Calculate silhouette scores
        calculated_scores.score(X = X, y = labels)

def test_realCluster():
    # Check if silhouette work in real cluster and has the same output result with sklearn
    t_clusters, t_labels = make_clusters(k=4)

    new = KMeans(4)
    new.fit(t_clusters)
    labels = new.predict(t_clusters)

    calculated_scores = silhouette.Silhouette()
    calculated_scores.score(X = t_clusters, y = labels)
    sci_score = silhouette_score(t_clusters, labels)

    np.testing.assert_allclose(calculated_scores.outscore, sci_score, atol=0.1)


