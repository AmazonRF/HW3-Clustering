# Write your k-means unit tests here
import pytest
from cluster import KMeans 

def test_invalid_k():
    with pytest.raises(ValueError) as e:
        KMeans(k='invalid', tol=1e-4, max_iter=100)
    assert str(e.value) == "k must be a positive integer."

def test_negative_k():
    with pytest.raises(ValueError) as e:
        KMeans(k=-1, tol=1e-4, max_iter=100)
    assert str(e.value) == "k must be a positive integer."

def test_invalid_tol():
    with pytest.raises(ValueError) as e:
        KMeans(k=3, tol='invalid', max_iter=100)
    assert str(e.value) == "tol must be a positive float."

def test_negative_tol():
    with pytest.raises(ValueError) as e:
        KMeans(k=3, tol=-1.0, max_iter=100)
    assert str(e.value) == "tol must be a positive float."

def test_invalid_max_iter():
    with pytest.raises(ValueError) as e:
        KMeans(k=3, tol=1e-4, max_iter='invalid')
    assert str(e.value) == "max_iter must be a positive integer."

def test_negative_max_iter():
    with pytest.raises(ValueError) as e:
        KMeans(k=3, tol=1e-4, max_iter=-100)
    assert str(e.value) == "max_iter must be a positive integer."
