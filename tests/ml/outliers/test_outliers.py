import unittest

import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, KMeans
from sklearn.covariance import EmpiricalCovariance, MinCovDet
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.mixture import GaussianMixture

from dsbox.ml.outliers import CovarianceOutliers, GaussianProcessOutliers
from dsbox.ml.outliers import GMMOutliers, ClusteringOutliers
from dsbox.ml.outliers import KMeansOneClusterOutliers, KMeansIterativeOneClusterOutliers
from dsbox.ml.outliers import MADOutliers, FFTOutliers


class CovarianceOutliersTest(unittest.TestCase):
    def test_covarianceoutliers_constructor_should_accept_different_scikit_covariance_estimators(self):
        # given
        robust_cov = MinCovDet()
        emp_cov = EmpiricalCovariance()

        # when
        cov_outliers_1 = CovarianceOutliers(emp_cov)
        cov_outliers_2 = CovarianceOutliers(robust_cov)

        # then
        self.assertTrue(isinstance(cov_outliers_1, CovarianceOutliers))
        self.assertTrue(isinstance(cov_outliers_2, CovarianceOutliers))

    def test_covarianceoutliers_predict_proba_gives_biggest_proba_to_biggest_outlier(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        cov_outliers = CovarianceOutliers()
        cov_outliers.fit(df)
        probas = cov_outliers.predict_proba(df)
        outlier_index = np.argmax(probas)

        # then
        outlier_index_true = 6
        self.assertEqual(outlier_index_true, outlier_index)

    def test_covarianceoutliers_predict_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        cov_outliers = CovarianceOutliers()
        cov_outliers.fit(df)
        outliers = cov_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, False, False, True, True, False, False, False,
                         False, False, False]
        self.assertListEqual(outliers_true, outliers.tolist())


class GaussianProcessOutliersTest(unittest.TestCase):
    def test_gpoutliers_predict_should_return_correct_values(self):
        # given
        data = np.random.random_sample(1000) * 2 - 1
        data[300] = 5
        data[700] = -6
        df = pd.DataFrame(data)

        # when
        gp_outliers = GaussianProcessOutliers(GaussianProcessRegressor(alpha=0.9, normalize_y=True), n_samples=100)
        gp_outliers.fit(df)
        outliers = gp_outliers.predict(df, confidence=0.999)
        outlier_positions = np.argwhere(outliers == np.amax(outliers)).flatten().tolist()

        # then
        outlier_positions_true = [300, 700]
        self.assertListEqual(outlier_positions_true, outlier_positions)


class KMeansOneClusterOutliersTest(unittest.TestCase):
    def test_kmeansonecluster_outliers_predict_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        kmoc_outliers = KMeansOneClusterOutliers()
        kmoc_outliers.fit(df)
        outliers = kmoc_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, False, False, True, True, False, False, False,
                         False, False, False]
        self.assertListEqual(outliers_true, outliers.tolist())


class KMeansIterativeOneClusterOutliersTest(unittest.TestCase):
    def test_kmeans_iterative_onecluster_outliers_predict_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        kmoc_outliers = KMeansIterativeOneClusterOutliers()
        outliers = kmoc_outliers.fit_predict(df)

        # then
        outliers_true = [2., 1., 1., 2., 2., -1., 0., 0., 1., 1., -1., 1., 1., 2.]
        self.assertListEqual(outliers_true, outliers.tolist())


class GMMOutliersTest(unittest.TestCase):
    def test_gmm_outliers_predict_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        gmm_outliers = GMMOutliers()
        gmm_outliers.fit(df)
        outliers = gmm_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, False, False, True, True, False, False, False,
                         False, False, False]
        self.assertListEqual(outliers_true, outliers.tolist())


class ClusteringOutliersTest(unittest.TestCase):
    def test_clustering_outliers_predict_proba_with_unclustered_strategy_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=DBSCAN(min_samples=2), strategy='unclustered')
        clustering_outliers.fit(df)
        outliers = clustering_outliers.predict_proba(df)

        # then
        outliers_true = [0., 0., 0., 0., 1., 0., 1., 1., 1., 1., 0., 0.,
                         1., 0.]
        self.assertListEqual(outliers_true, outliers.tolist())

    def test_clustering_outliers_predict_with_unclustered_strategy_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=DBSCAN(min_samples=2), strategy='unclustered')
        clustering_outliers.fit(df)
        outliers = clustering_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, True, False, True, True, True, True, False, False,
                         True, False]
        self.assertListEqual(outliers_true, outliers.tolist())

    def test_clustering_outliers_predict_proba_with_frontier_strategy_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=KMeans(n_clusters=2), strategy='frontier')
        clustering_outliers.fit(df)
        outliers_probas = clustering_outliers.predict_proba(df)

        # then
        outliers_true = [0.01861993, 0.02190581, 0.02190581, 0.01861993, 0.0109529, 0.01533406
            , 0.00196078, 0.00196078, 0.08324206, 0.03066813, 0.01533406, 0.02190581
            , 0.02738226, 0.01861993]
        self.assertListEqual(outliers_true, np.round(outliers_probas, 8).tolist())

    def test_clustering_outliers_predict_with_frontier_strategy_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=KMeans(n_clusters=2), strategy='frontier')
        clustering_outliers.fit(df)
        outliers = clustering_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, False, False, False, False, True, True, False, False,
                         True, False]
        self.assertListEqual(outliers_true, outliers.tolist())

    def test_clustering_outliers_predict_proba_with_size_strategy_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=KMeans(n_clusters=2), strategy='size')
        clustering_outliers.fit(df)
        outliers_probas = clustering_outliers.predict_proba(df)

        # then
        outliers_true = [0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714, 0.14285714
            , 0.85714286, 0.85714286, 0.14285714, 0.14285714, 0.14285714, 0.14285714
            , 0.14285714, 0.14285714]
        self.assertListEqual(outliers_true, np.round(outliers_probas, 8).tolist())

    def test_clustering_outliers_predict_with_size_strategy_and_kmeans_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=KMeans(n_clusters=2), strategy='size')
        clustering_outliers.fit(df)
        outliers = clustering_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, False, False, True, True, False, False, False,
                         False, False, False]
        self.assertListEqual(outliers_true, outliers.tolist())

    def test_clustering_outliers_predict_with_size_strategy_and_gmm_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=GaussianMixture(n_components=2), strategy='size')
        clustering_outliers.fit(df)
        outliers = clustering_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, False, False, True, True, False, False, False,
                         False, False, False]
        self.assertListEqual(outliers_true, outliers.tolist())

    def test_clustering_outliers_predict_with_size_strategy_and_dbscan_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=DBSCAN(min_samples=2), strategy='size',
                                                 threshold=0.8)
        clustering_outliers.fit(df)
        outliers = clustering_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, False, True, False, False, False, False, True, False
            , False, False]
        self.assertListEqual(outliers_true, outliers.tolist())

    def test_clustering_outliers_predict_proba_with_density_strategy_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=KMeans(n_clusters=2), strategy='density')
        clustering_outliers.fit(df)
        outliers_probas = clustering_outliers.predict_proba(df)

        # then
        outliers_true = [0.26737475, 0.26737475, 0.26737475, 0.26737475, 0.26737475, 0.26737475
            , 0.73262525, 0.73262525, 0.26737475, 0.26737475, 0.26737475, 0.26737475
            , 0.26737475, 0.26737475]
        self.assertListEqual(outliers_true, np.round(outliers_probas, 8).tolist())

    def test_clustering_outliers_predict_with_density_strategy_and_dbscan_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        clustering_outliers = ClusteringOutliers(cluster_estimator=DBSCAN(min_samples=2), strategy='density',
                                                 threshold=0.8)
        clustering_outliers.fit(df)
        outliers = clustering_outliers.predict(df)

        # then
        outliers_true = [False, False, False, False, True, False, True, True, True, True, False, False
            , True, False]
        self.assertListEqual(outliers_true, outliers.tolist())


class WrappersTest(unittest.TestCase):
    def test_mad_outliers_predict_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        mad_outliers = MADOutliers(threshold=0.9)
        outliers = mad_outliers.fit_predict(df)

        # then
        outliers_true = [False, False, False, False, True, False, True, True, True, True, False, False
            , True, False]

        self.assertListEqual(outliers_true, outliers.tolist())

    def test_fft_outliers_predict_should_return_correct_values(self):
        # given
        df = pd.DataFrame([1, 0, 0, 1, 10, 2, 115, 110, 32, 16, 2, 0, 15, 1])

        # when
        fft_outliers = FFTOutliers(threshold=0.9)
        outliers = fft_outliers.fit_predict(df)

        # then
        outliers_true = [False, False, False, False, False, False, False, True, False, False, False, False
            , False, False]

        self.assertListEqual(outliers_true, outliers.tolist())


if __name__ == '__main__':
    unittest.main()
