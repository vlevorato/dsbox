import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from mpl_toolkits.mplot3d import Axes3D

__author__ = "Vincent Levorato"
__credits__ = "Madalina Ciortan"
__source__ = "https://towardsdatascience.com/simple-example-of-2d-density-plots-in-python-83b83b934f67"
__license__ = "Apache 2.0"


def plot_gaussian_density(X, dimensions=2):
    """
    Plot a 2d gaussian 
    
    Parameters
    ----------
    X: array-like (2D)
        data used to estimate density
    
    dimensions: int, optional, (default=2)
        plot dimension of gaussian densities
    
    Examples
    --------
    
    >>> from dsbox.ml.visualization.distribution import plot_gaussian_density
    >>> from sklearn.datasets import make_blobs
    
    >>> n_components = 3
    >>> X , _ = make_blobs(n_samples=300, centers=n_components, cluster_std = [2, 1.5, 1], random_state=42)
    
    >>> plot_gaussian_density(X, dimensions=2)
    >>> plot_gaussian_density(X, dimensions=3)

    """

    # Extract x and y
    x = X[:, 0]
    y = X[:, 1]
    # Define the borders
    deltaX = (np.max(x) - np.min(x)) / 10
    deltaY = (np.max(y) - np.min(y)) / 10
    xmin = np.min(x) - deltaX
    xmax = np.max(x) + deltaX
    ymin = np.min(y) - deltaY
    ymax = np.max(y) + deltaY

    # Create meshgrid
    xx, yy = np.mgrid[xmin:xmax:100j, ymin:ymax:100j]

    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([x, y])
    kernel = st.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    if dimensions == 2:
        fig = plt.figure(figsize=(8, 8))
        ax = fig.gca()
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        cfset = ax.contourf(xx, yy, f, cmap='coolwarm')
        ax.imshow(np.rot90(f), cmap='coolwarm', extent=[xmin, xmax, ymin, ymax])
        cset = ax.contour(xx, yy, f, colors='k')
        ax.clabel(cset, inline=1, fontsize=10)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        plt.title('2D Gaussian Kernel density estimation')
        plt.show()

    if dimensions == 3:
        fig = plt.figure(figsize=(13, 7))
        ax = Axes3D(fig)
        surf = ax.plot_surface(xx, yy, f, rstride=1, cstride=1, cmap='coolwarm', edgecolor='none')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('PDF')
        ax.set_title('Surface plot of Gaussian 2D KDE')
        fig.colorbar(surf, shrink=0.5, aspect=5)  # add color bar indicating the PDF
        ax.view_init(60, 35)
        plt.show()
