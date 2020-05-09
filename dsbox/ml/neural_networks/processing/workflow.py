from dsbox.ml.neural_networks import KerasFactory
from dsbox.ml.neural_networks.processing import Text2Sequence
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class TextNeuralNetPipeline(BaseEstimator, ClassifierMixin):
    """
    Pipeline used to fit/predict raw text in a classification problem context. It transforms the text data
    via Text2Sequence transformer and passes the result to a keras model.
    
    Parameters
    ----------
    text2seq : object (default=Text2Sequence())
        transformer used to transform text into integer sequences.
        
    factory_class : KerasFactory (default=KerasFactory)
        set the factory used to build the keras model. By default, it takes an abstract class, which has to 
        be replaced by a concrete one (LSTMFactory or CNN_LSTMFactory for instance).
        
    num_labels : int, optional (default=2)
        amount of class labels
        
    Attributes
    ----------
    model_ : KerasClassifier
        keras scikit wrapper containing original keras architecture model.
    
            
    Examples
    --------
    >>> import numpy as np
    >>> np.random.seed(42) # for making doctests stable
    >>> from dsbox.ml.neural_networks.keras_factory.text_models import LSTMFactory
    >>> from dsbox.ml.neural_networks.processing.workflow import TextNeuralNetPipeline
    
    >>> x_train = np.array([u'this is really really awesome !', \
                            u'it is so awesome !', \
                            u'that sucks'] \
                           )
    >>> y_train = np.array([1, 1, 0])

    >>> model = TextNeuralNetPipeline(factory_class=LSTMFactory, num_labels=2, random_state=42)
    >>> _ = model.fit(x_train, y_train, verbose=0) # to avoid doctests error

    >>> x_test = np.array([u'it is really awesome !'])
    >>> model.predict(x_test)
    array([1])
    
    """
    def __init__(self, text2seq=Text2Sequence(), factory_class=KerasFactory, num_labels=2, random_state=42):
        self.text2seq = text2seq
        self.factory = factory_class()
        self.num_labels = num_labels
        self.random_state = random_state

        self.attr_to_check = ["model_"]

    def fit(self, X, y, **kwargs):
        """
        Fit the workflow by building the word corpus, and fitting the keras model.
    
        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str
        y : array-like, shape (n_samples,)
            Class targets.
        **kwargs : 
            parameters passed to inner keras model
    
        Returns
        -------
        self : object
            Returns an instance of self.
        """

        x = self.text2seq.fit_transform(X)
        y_enc = np_utils.to_categorical(y, self.num_labels)

        self.model_ = KerasClassifier(build_fn=self.factory.create_model,
                                      dictionary_size=self.text2seq.dictionary_size_,
                                      num_labels=self.num_labels)

        self.model_.fit(x, y_enc, **kwargs)

        return self

    def predict(self, X):
        """
        Predict a list of texts to belong to a known class.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        numpy array of class ids.

        """
        check_is_fitted(self, self.attr_to_check)

        x = self.text2seq.transform(X)
        return self.model_.predict(x)

    def predict_proba(self, X):
        """
        Predict a list of texts to belong to a known class.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        probability of belonging to each class id.
        """

        check_is_fitted(self, self.attr_to_check)

        x = self.text2seq.transform(X)
        return self.model_.predict_proba(x)
