import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, _tree

__author__ = "Romain Ayres, Veltin Dupont, Matthieu Lagacherie, Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class FeatureContributions(BaseEstimator):
    """
    A wrapper for all the action needed to extract the feature importance
    out of the tree interpreter, and plot the results.

        Parameters
        ----------
        model: DecisionTreeRegressor, DecisionTreeClassifier or RandomForestRegressor, RandomForestClassifier
            Scikit-learn model on which the prediction should be decomposed.
            
        estimator_type: {'classification', 'regression' } optional, default 'classification'
            Specify estimator type.
        
        References
        ----------
        [1]  https://github.com/andosa/treeinterpreter
        BSD 3-clause "New" or "Revised" License
    """

    ESTIMATOR_TYPES = {'classification': 0, 'regression': 1}

    def __init__(self, model, estimator_type='classification'):
        self.model = model
        self.Xtest = None
        self.ytest = None
        self.ypred = None
        self.contributions = None

        if estimator_type not in self.ESTIMATOR_TYPES.keys():
            raise ValueError("estimator_type parameter must belong to values:" + self.ESTIMATOR_TYPES.keys())

        self.estimator_type = self.ESTIMATOR_TYPES[estimator_type]

    def predict(self, X):
        """
        Computes the weights given by the tree interpreter.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Data used by tree interpreter

        Returns
        -------

        """
        # """Computes the weights given by the tree interpreter."""

        predictions, bias, contributions = _TreeInterpreter.predict(self.model, X)
        self._save_predictions(predictions)
        self._format_and_save_contributions(contributions, predictions, X)

        return predictions, bias, contributions

    def _format_and_save_contributions(self, contributions, predictions, dataframe):
        contrib = self._extract_significant_contributions(contributions, predictions)
        formatted_contrib = self._format_contributions_according_to_dataframe_format(contrib, dataframe)
        self._save_contributions(formatted_contrib)

    def _extract_significant_contributions(self, contributions, predictions):
        if self.estimator_type == self.ESTIMATOR_TYPES['classification']:
            contrib = [self._squeeze_class(c, p) for c, p in zip(contributions, predictions)]
        else:
            contrib = contributions
        return contrib

    def _save_predictions(self, predictions):
        self.ypred = predictions

    def _save_contributions(self, contributions):
        self.contributions = contributions

    @staticmethod
    def _format_contributions_according_to_dataframe_format(contributions, dataframe):
        if type(dataframe) == pd.DataFrame:
            new_contributions = pd.DataFrame(contributions, index=dataframe.index, columns=dataframe.columns)
        else:
            new_contributions = contributions
        return new_contributions

    @staticmethod
    def _squeeze_class(contributions, prediction):
        """Reduces the dimension of the tree interpreter output by selecting
        the weight of the predicted category only.
        Ex: if contributions = [[-1, 2, -1],
                                [-2, 5, -3]]
            and prediction = [0.1, 0.9, 0.],
            then it returns [2, 5].

        N.B.: This works well for multi class, but seems dumb for binary classication.

        TODO: implement a fixed column extraction for the binary case.
        """
        pred = np.argmax(prediction)
        return contributions[:, pred]


class _TreeInterpreter:
    @classmethod
    def _get_tree_paths(self, tree, node_id, depth=0):
        """
        Returns all paths through the tree as list of node_ids
        """
        if node_id == _tree.TREE_LEAF:
            raise ValueError("Invalid node_id %s" % _tree.TREE_LEAF)

        left_child = tree.children_left[node_id]
        right_child = tree.children_right[node_id]

        if left_child != _tree.TREE_LEAF:
            left_paths = self._get_tree_paths(tree, left_child, depth=depth + 1)
            right_paths = self._get_tree_paths(tree, right_child, depth=depth + 1)

            for path in left_paths:
                path.append(node_id)
            for path in right_paths:
                path.append(node_id)
            paths = left_paths + right_paths
        else:
            paths = [[node_id]]

        return paths

    @classmethod
    def _predict_tree(self, model, X):
        """
        For a given DecisionTreeRegressor or DecisionTreeClassifier,
        returns a triple of [prediction, bias and feature_contributions], such
        that prediction ≈ bias + feature_contributions.
        """

        leaves = model.apply(X)
        paths = self._get_tree_paths(model.tree_, 0)

        # remove the single-dimensional inner arrays
        values = model.tree_.value.squeeze()

        # check if values have only one axis
        if len(values.shape) == 1:
            values = np.array([values])

        if type(model) == DecisionTreeRegressor:
            line_shape = X.shape[1]
        elif type(model) == DecisionTreeClassifier:
            # scikit stores category counts, we turn them into probabilities
            normalizer = values.sum(axis=1)[:, np.newaxis]
            normalizer[normalizer == 0.0] = 1.0
            values /= normalizer

            line_shape = (X.shape[1], model.n_classes_)

        leaves_set = set(leaves)
        paths_to_compute = [path for path in paths if path[0] in leaves_set]

        contribs = {}
        for _, path in enumerate(paths_to_compute):
            path.reverse()
            leaf = path[-1]
            contribs[leaf] = np.zeros(line_shape)
            for i in range(len(path) - 1):
                contrib = values[path[i + 1]] - \
                          values[path[i]]
                contribs[leaf][model.tree_.feature[path[i]]] += contrib

        contributions = [contribs[leaf] for leaf in leaves]
        direct_prediction = values[leaves]
        biases = [values[path[0]]] * len(X)

        return direct_prediction, biases, contributions

    @classmethod
    def _predict_forest(self, model, X):
        """
        For a given RandomForestRegressor or RandomForestClassifier,
        returns a triple of [prediction, bias and feature_contributions], such
        that prediction ≈ bias + feature_contributions.
        """
        biases = []
        contributions = []
        predictions = []

        for tree in model.estimators_:
            pred, bias, contribution = self._predict_tree(tree, X)
            biases.append(bias)
            contributions.append(contribution)
            predictions.append(pred)
        return (np.mean(predictions, axis=0), np.mean(biases, axis=0),
                np.mean(contributions, axis=0))

    @classmethod
    def predict(self, model, X):
        """ Returns a triple (prediction, bias, feature_contributions), such
        that prediction ≈ bias + feature_contributions.
        
        Parameters
        ----------
        model : DecisionTreeRegressor, DecisionTreeClassifier or
            RandomForestRegressor, RandomForestClassifier
        Scikit-learn model on which the prediction should be decomposed.
        X : array-like, shape = (n_samples, n_features)
        Test samples.
        
        Returns
        -------
        decomposed prediction : triple of
        * prediction, shape = (n_samples) for regression and (n_samples, n_classes)
            for classification
        * bias, shape = (n_samples) for regression and (n_samples, n_classes) for
            classification
        * contributions, shape = (n_samples, n_features) for regression or
            shape = (n_samples, n_features, n_classes) for classification
        """
        # Only single out response variable supported,
        if model.n_outputs_ > 1:
            raise ValueError("Multilabel classification trees not supported")

        if (type(model) == DecisionTreeRegressor or
                    type(model) == DecisionTreeClassifier):
            return self._predict_tree(model, X)
        elif (type(model) == RandomForestRegressor or
                      type(model) == RandomForestClassifier):
            return self._predict_forest(model, X)
        else:
            raise ValueError("Wrong model type. Base learner needs to be \
                DecisionTreeClassifier or DecisionTreeRegressor.")
