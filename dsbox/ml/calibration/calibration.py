import numpy as np
from dsbox.ml.utils import check_estimator_predict_proba

from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class BinaryCalibrator(BaseEstimator, ClassifierMixin):
    """ Probability calibration using Bayes Minimum Risk theory to calibrate prediction 
    with unbalanced binary target.
    
    It is known that undersampling one class modifies the priors of the training set and 
    consequently biases the posterior probabilities of a classifier. This class finds the
    correct classification threshold and how to adjust it after undersampling to keep
    unbiased posterior prediction probabilities.
    
    Remark: this method concerns the probabilities of prediction for the underrepresented class (p').
    Probabilities for the major class are just computed as 1 - p'.
            

        Parameters
        ----------
        base_estimator : BaseEstimator
            estimator to calibrate
            
        major_class_proportion: float, optional, default 1.0
            Proportion of the major class to consider for undersampling.
            If major_class_proportion == 1, then the balance will be 50/50. To undersample less, reduce 
            this parameter.
       
        References
        ----------
        .. [1]  Andrea Dal Pozzolo and al. Calibrating Probability with Undersampling for 
                Unbalanced Classification. IEEE Symposium Series on Computational Intelligence, 
                SSCI 2015, 159-166

        See also
        --------
        sklearn.calibration.CalibratedClassifierCV : Probability calibration with isotonic regression or sigmoid
        
    """

    def __init__(self, base_estimator, major_class_proportion=1.0):

        self.base_estimator = base_estimator

        check_estimator_predict_proba(self.base_estimator)

        self.major_class_value = 0
        self.minor_class_value = 1

        self.major_class_proportion = major_class_proportion

        self.attr_to_check = ["major_class_amount_",
                              "minor_class_amount_",
                              "beta_",
                              "tau_prime_",
                              "X_",
                              "y_",
                              "classes_"]

    def fit(self, X, y):
        """ Fit the calibrated model
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
            
        y : array-like, shape (n_samples,)
            Target values.

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        check_X_y(X, y)

        unique_y = set(y)

        if len(unique_y) != 2:
            raise ValueError('BinaryCalibrator can only fit binary class, y modalities = ' + str(len(unique_y)))
        if 0 not in unique_y or 1 not in unique_y:
            raise ValueError('Target class value must be 0 and 1.')

        self.classes_ = list(unique_y)

        # check which class is in majority
        np_y = np.array(y).astype('int')
        val_distrib = np.bincount(np_y)
        if val_distrib[0] < val_distrib[1]:
            self.major_class_value = 1
            self.minor_class_value = 0

        # get amount of elements for each class
        self.major_class_amount_ = val_distrib[self.major_class_value]
        self.minor_class_amount_ = val_distrib[self.minor_class_value]

        # compute beta: ratio of the prior probabilities
        self.beta_ = (self.minor_class_amount_ / float(len(y))) / (self.major_class_amount_ / float(len(y)))

        # shuffle major class indexes for undersampling

        self.X_ = X.copy()
        self.X_['y'] = y
        self.X_['y'] = self.X_['y'].astype('int')
        X_major_class_indexes = np.array(self.X_[self.X_['y'] == self.major_class_value].index)
        np.random.shuffle(X_major_class_indexes)

        # remove data from major class to balance the dataset
        amount_index_to_remove = int((1 - self.beta_) * (self.major_class_amount_ * self.major_class_proportion))
        index_to_remove = X_major_class_indexes[0:amount_index_to_remove]
        df_X_to_remove = self.X_.iloc[index_to_remove]
        self.X_ = self.X_.drop(df_X_to_remove.index)
        self.y_ = self.X_['y']

        # tau_prime: threshold that guarantees equal accuracy after the posterior probability correction
        self.tau_s_ = len(self.X_[self.X_['y'] == self.major_class_value]) / float(len(self.X_))
        self.tau_prime_ = (self.beta_ * self.tau_s_) / ((self.beta_ - 1) * self.tau_s_ + 1)

        del self.X_['y']

        self.base_estimator.fit(self.X_, self.y_)

        return self

    def predict_proba(self, X):
        """Posterior probabilities of classification
        This function returns posterior probabilities of classification
        according to beta parameter computed during fit
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.
            
        Returns
        -------
        y_probas : array, shape (n_samples, n_classes)
            The predicted probas.
        """
        check_is_fitted(self, self.attr_to_check)

        check_array(X)

        y_probas = self.base_estimator.predict_proba(X)
        for i in range(0, len(y_probas)):
            yp = y_probas[i]
            yp[self.minor_class_value] = self._bias_corrected_probability(yp[self.minor_class_value])
            yp[self.major_class_value] = 1 - yp[self.minor_class_value]
            y_probas[i] = yp  # should be useless as yp is a reference

        return y_probas

    def predict(self, X):
        """Predict the target of new samples. Can be different from the
        prediction of the uncalibrated classifier.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The samples.
            
        Returns
        -------
        C : array, shape (n_samples,)
            The predicted class.
        """

        y_pred_probas = self.predict_proba(X)
        y_pred = []
        for yp in y_pred_probas:
            if yp[1] >= 0.5:
                y_pred.append(1)
            else:
                y_pred.append(0)

        return y_pred

    def _bias_corrected_probability(self, p):

        return (self.beta_ * p) / ((self.beta_ * p) - p + 1)

    def bias_corrected_threshold(self):
        """
        Returns the threshold that guarantees equal accuracy after the posterior probability correction.
        
        Returns
        -------
        tau_prime_ : float
            tau' threshold
        """

        return self.tau_prime_
