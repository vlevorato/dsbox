from collections import Counter

import numpy as np
import pandas as pd
from dsbox.ml.outliers import fft_outliers, mad_outliers
from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.exceptions import NotFittedError
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import BayesianGaussianMixture
from sklearn.mixture.base import BaseMixture
from sklearn.neighbors.kde import KernelDensity
from sklearn.utils.validation import check_is_fitted, column_or_1d

from scipy.stats import norm

__author__ = "Vincent Levorato, Rémy Frenoy"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class CovarianceOutliers(BaseEstimator):
    """ Covariance wrapper outlier estimator.
    
    Encapsulates sklearn covariance estimator to extract outliers from Mahalanobis squared distance.
    
    Parameters
    ----------
    
    cov_estimator : EmpiricalCovariance, optional (default=MinCovDet())
    
    threshold : float (default=None)
        Used by predict method : if probability returned by predict_proba method is above this value, the element 
        is considered as an outlier. If not set, it takes the mean of probabilities.
    
    
    Attributes
    ----------
    
    mahal_dist_ : array
       Mahalanobis squared distances computed by the covariance estimator. 
       
       
    Examples
    --------
    >>> import pandas as pd
    >>> from sklearn.covariance import MinCovDet
    >>> from dsbox.ml.outliers import CovarianceOutliers
    
    >>> X = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])
    >>> cov_outliers = CovarianceOutliers(MinCovDet())
    >>> cov_outliers.fit(X)
    CovarianceOutliers(cov_estimator=MinCovDet(assume_centered=False, random_state=None, store_precision=True,
         support_fraction=None),
              threshold=None)
    >>> outliers = cov_outliers.predict(X)
    >>> outliers.values
    array([False, False, False, False, False, False,  True,  True, False,
           False, False, False, False, False])
    
    """

    def __init__(self, cov_estimator=MinCovDet(), threshold=None):
        if not isinstance(cov_estimator, EmpiricalCovariance):
            raise TypeError("Estimator must be a sklearn.covariance.EmpiricalCovariance class")

        self.cov_estimator = cov_estimator
        self.threshold = threshold

        self.attr_to_check = ["mahal_dist_"]

    def fit(self, X, y=None):
        """
        Fits the covariance model according to the given training data and parameters.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
            Returns self.
        """
        self.cov_estimator.fit(X)
        self.mahal_dist_ = self.cov_estimator.mahalanobis(X)

        return self

    def predict_proba(self, X=None):
        """
        Returns probability for each element to be an outlier. 
        Exactly, it computes the ratio of the Mahalanobis squared distance relative to the distance sum.
        
        Parameters
        ----------
        X : not used, present for API consistence purpose.

        Returns
        -------
        Probabilities for each element to be an outlier

        """
        check_is_fitted(self, self.attr_to_check)

        return self.mahal_dist_ / np.sum(self.mahal_dist_)

    def predict(self, X=None):
        """
        Returns a boolean tag for each element to be an outlier. It takes each Mahalanobis squared distance,
        transforms it in probability, and checks if it exceeds the threshold attribute.
        
        Parameters
        ----------
        X : not used, present for API consistence purpose.

        Returns
        -------
        Boolean array with outlier tag
        """
        probas = self.predict_proba(X)

        if self.threshold is None:
            self.threshold = np.mean(probas)
        return probas > self.threshold


class GaussianProcessOutliers(BaseEstimator):
    """ Gaussian Process wrapper outlier estimator

    Encapsulates sklearn Gaussian Process Regressor estimator to extract outliers which should be
    out of the confidence intervals along the "infinite" normal marginal distributions estimated by the
    GP estimator.

    Parameters
    ----------

    gp_estimator : GaussianProcessRegressor, optional (default=GaussianProcessRegressor(alpha=0.9, normalize_y=True))

    n_samples : int, optional (default=500)
        Corresponds to the amount of points used for the prediction (should be less than the original amount of elements)
        
    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> from sklearn.gaussian_process import GaussianProcessRegressor
    >>> from dsbox.ml.outliers import GaussianProcessOutliers

    >>> np.random.seed(42)
    >>> data = np.random.random_sample(200) * 2 - 1
    >>> data[150] = 5
    >>> data[190] = -6
    >>> df = pd.DataFrame(data)
    
    >>> gp_outliers = GaussianProcessOutliers(GaussianProcessRegressor(alpha=0.95, normalize_y=True), n_samples=100)
    >>> gp_outliers.fit(df)
    GaussianProcessOutliers(gp_estimator=GaussianProcessRegressor(alpha=0.95, copy_X_train=True, kernel=None,
                 n_restarts_optimizer=0, normalize_y=True,
                 optimizer='fmin_l_bfgs_b', random_state=None),
                n_samples=100)
    >>> outliers = gp_outliers.predict(df, confidence=0.999)
    >>> outlier_positions = np.argwhere(outliers == np.amax(outliers)).flatten().tolist()
    >>> outlier_positions
    [150, 190]
    
    """

    def __init__(self, gp_estimator=GaussianProcessRegressor(alpha=0.9, normalize_y=True), n_samples=500):

        if not isinstance(gp_estimator, GaussianProcessRegressor):
            raise TypeError("Estimator must be a sklearn.gaussian_process.GaussianProcessRegressor class")

        self.gp_estimator = gp_estimator
        self.n_samples = n_samples

    def fit(self, X, y=None):
        """
        Fits the Gaussian Process estimator according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
            Training data, where n_samples is the number of samples
            and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
        """

        y = column_or_1d(X)
        x = np.arange(0, y.size)
        df_x = pd.DataFrame(x)
        self.gp_estimator.fit(df_x, y)

        return self

    def predict(self, X, confidence=0.9999):
        """
        Returns a boolean tag for each element to be an outlier. It uses the predict method of the 
        Gaussian Process estimator, smoothing the inputs value, and comparing the prediction according to
        confidence interval.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
        
        confidence : float, optional (default=0.9999)

        Returns
        -------
        Boolean array with outlier tag
        """
        y = column_or_1d(X)
        x = np.arange(0, y.size)

        if self.gp_estimator.normalize_y:
            y_origin_mean = 0
            y_origin_std = 1
        else:
            y_origin_mean = np.mean(y)
            y_origin_std = np.std(y)

        X_pred = pd.DataFrame(np.linspace(x.min(), x.max(), num=self.n_samples))
        y_mean, y_std = self.gp_estimator.predict(X_pred, return_std=True)
        tolerance = norm.ppf(confidence, y_origin_mean, y_origin_std)
        indices = [int(x) for x in np.linspace(0, 99, y.size)]

        lower_bound = y_mean[indices] - tolerance * y_std[indices]
        upper_bound = y_mean[indices] + tolerance * y_std[indices]

        outliers = (y < lower_bound) | (y > upper_bound)
        return outliers


class KMeansOneClusterOutliers(BaseEstimator):
    """ KMeans One Cluster Outliers estimator
    
    This estimator uses distance information to categorize outliers. In this case, KMeans algorithm is used,
    computing only one cluster and one centroid. Depending on the distance to the centroid, elements are categorized
    as outliers or not (the greater the distance of an element, the more likely it is to be considered an outlier).
    
    Parameters
    ----------
    
    kmeans_estimator : KMeans, optional (default=KMeans(n_clusters=1, n_jobs=-1))
    
    threshold : float (default=None)
        Used by predict method : if probability returned by predict_proba method is above this value, the element 
        is considered as an outlier. If not set, it takes the mean of probabilities.
    
    
    Examples
    --------
    >>> import pandas as pd
    >>> from dsbox.ml.outliers import KMeansOneClusterOutliers
    
    >>> df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])
    >>> kmoc_outliers = KMeansOneClusterOutliers()
    >>> kmoc_outliers.fit(df)
    KMeansOneClusterOutliers(kmeans_estimator=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=1, n_init=10, n_jobs=-1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0),
                 threshold=None)
    >>> outliers = kmoc_outliers.predict(df)
    >>> outliers
    array([False, False, False, False, False, False,  True,  True, False,
           False, False, False, False, False])
    
    """

    def __init__(self, kmeans_estimator=KMeans(n_clusters=1, n_jobs=-1), threshold=None):
        if not isinstance(kmeans_estimator, KMeans) and not isinstance(kmeans_estimator, MiniBatchKMeans):
            raise TypeError("Estimator must be a sklearn.cluster.KMeans or MiniBatchKMeans class")

        if kmeans_estimator.get_params()['n_clusters'] != 1:
            raise ValueError("KMeans estimator n_clusters parameter must be set to 1")

        self.kmeans_estimator = kmeans_estimator
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fits the kmeans one cluster model according to the given training data and parameters.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Training data, where n_samples is the number of samples
           and n_features is the number of features.
        y : not used, present for API consistence purpose.
        
        Returns
        -------
        self : object
           Returns self.
        """
        self.kmeans_estimator.fit(X)
        return self

    def predict_proba(self, X):
        """
        Returns probability for each element to be an outlier. 
        Exactly, it computes the ratio of the distance to the unique centroid with the distance sum.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Probabilities for each element to be an outlier

        """

        distances = self.kmeans_estimator.transform(X).ravel()
        sum_distances = np.sum(distances)
        if sum_distances == 0:
            return distances
        return distances / float(sum_distances)

    def predict(self, X):
        """
        Returns a boolean tag for each element to be an outlier. It takes each distance to the unique centroid,
        transform it in probability, and checks if it exceeds the threshold attribute.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Boolean array with outlier tag
        """
        probas = self.predict_proba(X)
        if self.threshold is None:
            self.threshold = np.mean(probas)
        return probas > self.threshold


class KMeansIterativeOneClusterOutliers(BaseEstimator):
    """ KMeans Iterative One Cluster Outliers estimator

    This estimator uses iteratively KMeans One Cluster Outliers estimator to find outliers. It applies n times
    the KMOC estimator, finding and removing outliers from the original dataset, giving them a tag which correspond
    to the iteration number they have been removed.

    Parameters
    ----------

    kmoc_estimator : KMeansOneClusterOutliers, optional (default=KMeansOneClusterOutliers())

    n_iterations : int, optional, (default=5)
        Max iterations amount used for finding outliers with KMOC estimator iteratively.

    Examples
    --------
    >>> import pandas as pd
    >>> from dsbox.ml.outliers import KMeansIterativeOneClusterOutliers
    
    >>> df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])
    >>> kmoc_outliers = KMeansIterativeOneClusterOutliers()
    >>> outliers = kmoc_outliers.fit_predict(df)
    >>> outliers
    array([ 2.,  1.,  1.,  2.,  2., -1.,  0.,  0.,  1.,  1., -1.,  1.,  1.,
            2.])

    """

    def __init__(self, kmoc_estimator=KMeansOneClusterOutliers(), n_iterations=5):
        if not isinstance(kmoc_estimator, KMeansOneClusterOutliers):
            raise TypeError("Estimator must be a bdacore.outliers.KMeansOneClusterOutliers class")

        self.kmoc_estimator = kmoc_estimator
        self.n_iterations = n_iterations

    def fit_predict(self, X, y=None):
        """
        Fits the kmeans one cluster model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Training data, where n_samples is the number of samples
           and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
           Returns self.
        """
        XX = X.copy()
        outlier_iter = np.ones(len(XX)) * -1
        i = 0

        while i < self.n_iterations and len(XX) > 0:
            self.kmoc_estimator.fit(XX)
            outliers = self.kmoc_estimator.predict(XX)
            if True in outliers:
                XX['outliers'] = outliers
                outlier_positions = XX[XX['outliers'] == True].index.values
                for position in outlier_positions:
                    outlier_iter[position] = i
                del XX['outliers']
                XX = XX.drop(outlier_positions)
                i += 1
            else:
                i = self.n_iterations

        return outlier_iter


class GMMOutliers(BaseEstimator):
    """ Gaussian mixture model Outliers estimator
    
    This estimator uses density information to categorize outliers. In this case, GMM algorithm is used,
    computing an amount of components. Depending on the PDF (probability density function), elements are categorized
    as outliers or not (the less density, the more likely it is to be considered an outlier).
    
    Parameters
    ----------
    
    gmm_estimator : BaseMixture, optional (default=BayesianGaussianMixture(n_components=1))
    
    threshold : float, optional(default=None)
        Used by predict method : if probability returned by predict_proba method is above this value, the element 
        is considered as an outlier. If not set, it takes the mean of probabilities.
        
    Examples
    --------
    
    >>> import pandas as pd
    >>> from dsbox.ml.outliers import GMMOutliers
    
    >>> df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])
    >>> gmm_outliers = GMMOutliers()
    >>> gmm_outliers.fit(df)
    GMMOutliers(gmm_estimator=BayesianGaussianMixture(covariance_prior=None, covariance_type='full',
                degrees_of_freedom_prior=None, init_params='kmeans',
                max_iter=100, mean_precision_prior=None, mean_prior=None,
                n_components=1, n_init=1, random_state=None, reg_covar=1e-06,
                tol=0.001, verbose=0, verbose_interval=10, warm_start=False,
                weight_concentration_prior=None,
                weight_concentration_prior_type='dirichlet_process'),
          threshold=None)
    >>> outliers = gmm_outliers.predict(df)
    >>> outliers
    array([False, False, False, False, False, False,  True,  True, False,
           False, False, False, False, False])
    
    """

    def __init__(self, gmm_estimator=BayesianGaussianMixture(n_components=1), threshold=None):
        if not isinstance(gmm_estimator, BaseMixture):
            raise TypeError("Estimator must be a sklearn.mixture.base.BaseMixture class")

        self.gmm_estimator = gmm_estimator
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fits the GMM model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Training data, where n_samples is the number of samples
           and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
           Returns self.
        """
        self.gmm_estimator.fit(X)
        return self

    def predict_proba(self, X):
        """
        Returns probability for each element to be an outlier. 
        It transforms probability densities obtained (score samples) by integrating them to 1 to probabilities
        of having this value. In this case, it returns probability for each element to be an outlier.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Probabilities for each element to be an outlier

        """

        dx = X.diff().abs().median().sum()
        return 1 - (np.exp(self.gmm_estimator.score_samples(X)) * np.power(dx, X.shape[1]))

    def predict(self, X):
        """
        Returns a boolean tag for each element to be an outlier. It takes probability density transformed
        into probability, and checks if it exceeds the threshold attribute.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Boolean array with outlier tag
        """
        probas = self.predict_proba(X)
        if self.threshold is None:
            self.threshold = np.mean(probas)
        return probas > self.threshold


class ClusteringOutliers(BaseEstimator):
    """ Clustering model Outliers estimator

    This estimator uses clustering information to categorize outliers. Depending on the strategy, this
    estimator can use un-clustering information (like DBSCAN which classify elements that not belongs to any
    cluster), frontier, size or density of the clusters.

    Parameters
    ----------

    cluster_estimator : BaseMixture or ClusterMixin, optional (default=DBSCAN())

    threshold : float, optional (default=None)
        Used by predict method : if probability returned by predict_proba method is above this value, the element 
        is considered as an outlier. If not set, it takes the mean of probabilities.
    
    strategy : {'unclustered', 'frontier', 'size', 'density'} (default='size')
        Stragegy used to identify outliers.
        
        'unclustered': if clustering estimator is able to put a label (-1) on elements out of
        all clusters (DBSCAN, MeanShift), this strategy will identify these elements as outliers.
        
        'frontier': this strategy considers, for each cluster, the farest elements of the center as outliers.
        
        'size': this strategy considers the elements of the smallest cluster(s) as outliers.
        
        'density': this strategy considers the elements of the less dense cluster(s) as outliers.
        
    kernel_density : KernelDensity
        Only used by 'density' strategy, used to compute density per cluster.
    
    Attributes
    ----------
    
    is_fitted : boolean
        Used only for consistency between clustering estimators methods.
    
    Examples
    --------
    
    >>> import pandas as pd 
    >>> from dsbox.ml.outliers import ClusteringOutliers
    
    >>> df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])
    >>> clustering_outliers = ClusteringOutliers(cluster_estimator=KMeans(n_clusters=2), strategy='size')
    >>> clustering_outliers.fit(df)
    ClusteringOutliers(cluster_estimator=KMeans(algorithm='auto', copy_x=True, init='k-means++', max_iter=300,
        n_clusters=2, n_init=10, n_jobs=1, precompute_distances='auto',
        random_state=None, tol=0.0001, verbose=0),
              kernel_density=KernelDensity(algorithm='auto', atol=0, bandwidth=0.2, breadth_first=True,
           kernel='epanechnikov', leaf_size=40, metric='euclidean',
           metric_params=None, rtol=0),
              strategy='size', threshold=None)
    >>> outliers = clustering_outliers.predict(df)
    >>> outliers
    array([False, False, False, False, False, False,  True,  True, False,
           False, False, False, False, False])
     
    
    """

    def __init__(self, cluster_estimator=DBSCAN(), threshold=None, strategy='size',
                 kernel_density=KernelDensity(kernel='epanechnikov', bandwidth=0.2)):
        if not isinstance(cluster_estimator, BaseMixture) and not isinstance(cluster_estimator, ClusterMixin):
            raise TypeError("Estimator must be a sklearn.mixture.base.BaseMixture class" +
                            " or a sklearn.base.ClusterMixin class.")

        self.cluster_estimator = cluster_estimator
        self.threshold = threshold
        self.strategy = strategy
        self.kernel_density = kernel_density
        self.is_fitted = False

    def fit(self, X, y=None):
        """
        Fits the clustering model according to the given training data and parameters.

        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Training data, where n_samples is the number of samples
           and n_features is the number of features.
        y : not used, present for API consistence purpose.

        Returns
        -------
        self : object
           Returns self.
        """
        self.cluster_estimator.fit(X)
        self.is_fitted = True
        return self

    def predict_proba(self, X):
        """
        Returns probability for each element to be an outlier, according to the chosen strategy.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Probabilities for each element to be an outlier

        """
        if not self.is_fitted:
            raise NotFittedError("This instance is not fitted yet.")

        labels = []
        if 'predict' in dir(self.cluster_estimator):
            labels = self.cluster_estimator.predict(X)
        else:
            labels = self.cluster_estimator.labels_

        probas = np.zeros(len(labels))

        if self.strategy == 'unclustered':
            return (labels == -1).astype('float')

        if self.strategy == 'frontier':
            distances = self.cluster_estimator.transform(X)
            probas_all = np.zeros(distances.shape)
            for col_id in range(0, distances.shape[1]):
                probas_all[:, col_id] = distances[:, col_id] / np.sum(distances[:, col_id])

            for i in range(0, len(labels)):
                probas[i] = probas_all[i, labels[i]]

        if self.strategy == 'size':
            X_size = float(len(X))
            counter = Counter(labels)
            size_ratio = np.zeros(len(counter.items()))
            for label in counter.keys():
                size_ratio[label] = counter[label] / X_size

            for i in range(0, len(labels)):
                probas[i] = 1 - size_ratio[labels[i]]

        if self.strategy == 'density':
            X['label_'] = labels
            score_label = {}
            for i in range(0, np.max(labels) + 1):
                self.kernel_density.fit(X[X['label_'] == i])
                score_label[i] = self.kernel_density.score(X[X['label_'] == i])

            del X['label_']

            # force the density for unclustered element to zero (if any)
            score_label[-1] = 0.0

            for i in range(0, len(labels)):
                probas[i] = 1 - (score_label[labels[i]] / sum(score_label.values()))

        return probas

    def predict(self, X):
        """
        Returns a boolean tag for each element to be an outlier. It takes predict_proba method, 
        and checks if it exceeds the threshold attribute.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Boolean array with outlier tag
        """
        probas = self.predict_proba(X)

        if self.strategy == 'unclustered':
            return probas == 1

        if self.threshold is None:
            self.threshold = np.mean(probas)
        return probas > self.threshold


class MADOutliers(BaseEstimator):
    """  Median Absolute Deviation outliers estimator
    
    A simple scikit wrapper using bdacore.outliers.utils.mad_outliers method. See documentation for
    further purpose.
    
    
    Parameters
    ----------
    
    cutoff : int, optional (default=2)
        amount of times residuals relative to the median exceed the ratio to the MAD
    
    
    Examples
    --------
    >>> import pandas as pd 
    >>> from dsbox.ml.outliers import MADOutliers
    
    >>> df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])
    >>> mad_outliers = MADOutliers()
    >>> outliers = mad_outliers.fit_predict(df)
    >>> outliers.values
    array([False, False, False, False,  True, False,  True,  True,  True,
            True, False, False,  True, False])
    
    
    """

    def __init__(self, cutoff=2, threshold=None):
        self.cutoff = cutoff
        self.threshold = threshold

    def fit(self, X, y=None):
        """
        Fits the MAD model according to the given training data and parameters.
        
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
          Training data, where n_samples is the number of samples
          and n_features is the number of features.
        y : not used, present for API consistence purpose.
        
        Returns
        -------
        self : object
          Returns self.
        """
        self.X_ = pd.DataFrame(columns=X.columns)
        for column in X.columns:
            self.X_[column] = mad_outliers(X[column], cutoff=self.cutoff)

        return self

    def predict_proba(self, X=None):
        """
        Returns probability for each element to be an outlier.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Probabilities for each element to be an outlier

        """

        return self.X_.mean(axis=1)

    def predict(self, X):
        """
        Returns a boolean tag for each element to be an outlier. It takes predict_proba method, 
        and checks if it exceeds the threshold attribute.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Boolean array with outlier tag
        """
        probas = self.predict_proba(X)

        if self.threshold is None:
            self.threshold = np.mean(probas)
        return probas > self.threshold

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)


class FFTOutliers(BaseEstimator):
    """  Fast Fourier Transformation outliers estimator

    A simple scikit wrapper using bdacore.outliers.utils.fft_outliers method. See documentation for
    further purpose.


    Parameters
    ----------

    freq_cut_index : int, optional (default=None)
        set the index in the FFT numpy array beyond which the frequency is considered to be filtered. The highest,
        the less filtering. By default, it sets this index to 0.9 of x length
        
    outlier_proportion : float, optional (default=0.1)
        set the proportion of outliers to return


    Examples
    --------
    >>> import pandas as pd 
    >>> from dsbox.ml.outliers import FFTOutliers

    >>> df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])
    >>> fft_outliers = FFTOutliers()
    >>> outliers = fft_outliers.fit_predict(df)
    >>> outliers.values
    array([False, False, False, False, False, False, False,  True, False,
           False, False, False, False, False])


    """

    def __init__(self, freq_cut_index=None, outlier_proportion=0.1, threshold=None):
        self.freq_cut_index = freq_cut_index
        self.outlier_proportion = outlier_proportion
        self.threshold = threshold

    def fit(self, X=None, y=None):
        """
        Fits the FFT model according to the given training data and parameters.
 
        Parameters
        ----------
        X : array-like, shape = [n_samples, n_features]
           Training data, where n_samples is the number of samples
           and n_features is the number of features.
        y : not used, present for API consistence purpose.
 
        Returns
        -------
        self : object
           Returns self.
        """

        self.X_ = pd.DataFrame(columns=X.columns)
        for column in X.columns:
            self.X_[column] = fft_outliers(X[column], freq_cut_index=self.freq_cut_index,
                                           outlier_proportion=self.outlier_proportion)

        return self

    def predict_proba(self, X=None):
        """
        Returns probability for each element to be an outlier.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Probabilities for each element to be an outlier

        """

        return self.X_.mean(axis=1)

    def predict(self, X):
        """
        Returns a boolean tag for each element to be an outlier. It takes predict_proba method, 
        and checks if it exceeds the threshold attribute.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        Boolean array with outlier tag
        """
        probas = self.predict_proba(X)

        if self.threshold is None:
            self.threshold = np.mean(probas)
        return probas > self.threshold

    def fit_predict(self, X):
        self.fit(X)
        return self.predict(X)
