#http://rasbt.github.io/mlxtend/
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# Loading Plotting Utilities
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import itertools
from mlxtend.plotting import plot_decision_regions
import numpy as np

import ipdb

def plot_xor():
    xx, yy = np.meshgrid(np.linspace(-3, 3, 50), np.linspace(-3, 3, 50))
    rng = np.random.RandomState(0)
    X = rng.randn(300, 2)
    y = np.array(np.logical_xor(X[:, 0] > 0, X[:, 1] > 0), dtype=int)

    gs = gridspec.GridSpec(2, 2)
    fig = plt.figure(figsize=(10, 8))

    ax = plt.subplot(gs[0, 0])
    plt.plot(X[np.where(y == 0), 0], X[np.where(y == 0), 1], 'ro')
    plt.plot(X[np.where(y == 1), 0], X[np.where(y == 1), 1], 'bo')
    plt.title('XOR')
    plt.show()

def plot_decision_boundary(X, y_actual, inference):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    zz = inference(np.c_[xx.ravel(), yy.ravel()])
    zz = zz.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, zz, cmap=plt.cm.Paired)
    plt.scatter(X[:, 0], X[:, 1], c=y_actual, cmap=plt.cm.Spectral)
    plt.show()
