from sklearn.base import BaseEstimator, TransformerMixin
import types

__author__ = "Vincent Levorato, Eric Biernat"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


def project_continuous_on_categorical(dataframe, cat_col, cont_col, operation='mean'):
    return dataframe.groupby(cat_col)[cont_col].transform(operation)


class CategoricalProjector(BaseEstimator, TransformerMixin):
    """
    Aggregate continuous variables on categorical ones.

    Parameters
    ----------   
        categories_to_project: list
            List of the columns to be used as category for the projection.
            
        continous_cols: list
            List of the columns to be used as continuous variables for the projection.
        
        operation: str or function, optional, (default='mean')
            Operation used for aggregation.
            
    Examples
    --------
    >>> import pandas as pd
    >>> from dsbox.ml.feature_engineering import CategoricalProjector
    
    >>> df = pd.DataFrame({'item': ['A', 'A', 'B', 'B', 'B'], \
                           'town': ['Paris', 'London', 'Paris', 'Paris', 'Roma'], \
                           'price': [0.0, 1.0, 2.0, 2.0, 1.0], \
                           'quantity': [3, 4, 1, 2, 1] \
                           })
    >>> cat_projector = CategoricalProjector(['item'], ['price', 'quantity'])
    >>> cat_projector.fit_transform(df)
      item  mean_price_per_item  mean_quantity_per_item
    0    A             0.500000                3.500000
    1    A             0.500000                3.500000
    2    B             1.666667                1.333333
    3    B             1.666667                1.333333
    4    B             1.666667                1.333333
    """

    def __init__(self, categories_to_project, continous_cols, operation='mean'):
        self.categories_to_project = categories_to_project
        self.continous_cols = continous_cols
        self.operation = operation

    def fit(self, X=None, y=None):
        """
        No-op.
        This method doesn't do anything. It exists purely for compatibility
        with the scikit-learn transformer API.

        Parameters
        ----------
            X: array-like
            y: array-like

        Returns
        -------
            self

        """

        return self

    def transform(self, X):
        """
       Transform a dataframe to build projection from continuous variables on categorical ones. Column
       names are named using the aggregation function name (i.e mean_xxx_per_xxx)

       Parameters
       ----------
           X: dataframe
               Input pandas dataframe.

       Returns
       -------
           X: dataframe 
               Dataframe with continuous on categorical projection

       """

        X_result = X[self.categories_to_project]

        operation_name = self.operation

        if isinstance(self.operation, types.FunctionType):
            operation_name = self.operation.__name__

        for cat_col in self.categories_to_project:
            for cont_col in self.continous_cols:
                X_result.loc[:, operation_name + '_{}_per_{}'.format(cont_col, cat_col)] = \
                    project_continuous_on_categorical(X, cat_col, cont_col, operation=self.operation)

        return X_result


class TagEncoder(BaseEstimator, TransformerMixin):
    """
    Alternative to sklearn LabelEncoder. The main difference is the transform operation is not crashing if
    an unknown value is encountered.

    Parameters
    ----------
        missing_value: int, optional, default: -1
            value used to encode unknown categories during the transform operation
            
    Examples
    --------
    >>> import pandas as pd
    >>> from dsbox.ml.feature_engineering import TagEncoder

    >>> df = pd.DataFrame({'item': ['A', 'A', 'B', 'B', 'B']})
    >>> tagencoder = TagEncoder()
    >>> tagencoder.fit_transform(df['item'])
    [0, 0, 1, 1, 1]
    """

    def __init__(self, missing_value=-1):
        self.dict_id2tag = {}
        self.dict_tag2id = {}
        self.missing_value = missing_value

    def fit(self, X, y=None):
        """
        Fit tag encoder
        
        Parameters
        ----------
        X : array-like of shape (n_samples,)
            Target values.
            
        Returns
        -------
        self : returns an instance of self.
        """
        self.id_num_ = 0
        self.dict_id2tag = {}
        self.dict_tag2id = {}

        for x in X:
            if x not in self.dict_tag2id:
                self.dict_tag2id[x] = self.id_num_
                self.dict_id2tag[self.id_num_] = x
                self.id_num_ += 1

        if self.missing_value >= 0 and self.id_num_ >= self.missing_value:
            raise ValueError("Missing value id category is equal to one of the id category.")

        return self

    def transform(self, X):
        """
        Transform labels to normalized encoding.
        
        Parameters
        ----------
        X : array-like of shape [n_samples]
           Target values.
        Returns
        -------
        y : array-like of shape [n_samples]
        """
        x_transform = []
        for x in X:
            if x not in self.dict_tag2id:
                x_transform.append(self.missing_value)
            else:
                x_transform.append(self.dict_tag2id[x])

        return x_transform
