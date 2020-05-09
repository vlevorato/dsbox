import numpy as np
import pandas as pd
from dsbox.ml.markov.histrogramdata import HistogramData
from sklearn.base import BaseEstimator
from sklearn.utils.validation import column_or_1d, check_is_fitted

__author__ = "Vincent Levorato, Michel Lutz"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class MarkovSequenceMiner(BaseEstimator):
    """ Simple Markov Chain model.

    In principle, when we observe a sequence of chance experiments, all of the past outcomes could
    influence our predictions for the next experiment : the outcome of a given experiment can affect 
    the outcome of the next experiment. This type of process is called a Markov chain.

    The process starts in one of the states of a sequence and moves successively from one state to 
    another. If the chain is currently in state 'A, then it moves to state 'B' at the next step with
    a probability denoted by p_AB. These probabilities are stored in a transition matrix.

    The MarkovSequenceMiner builds the transition matrix with the fit() method. It's possible to test
    another sequence to get the probabilities of having a new state knowing only the last one or to get 
    the probabilities of having a new state knowing all the previous ones. 

    Parameters
    ----------
    init_label : str, optional (default=None)
        Gives explicitly the first state of the sequence. If not set, the first state will be the first
        element seen during the fit process.

    warm_start : bool, optional (default=False)
        When set to ``True``, remember the previous states seen during the previous predict process
        when calling the predict process.

    equiproba : bool, optional (default=False)
        When set to ``True``, ignore the init_label given value, and gives to the init vector the same
        probabilites weight (i.e. if there are 3 possible states, it gives 1/3 chances to have a state
        for the starting configuration).

    Attributes
    ----------
    labels_ : array of all the states
        Corresponds to states seen during the fit process.

    inv_dict_labels_ : dict
        For a given state, returns its position in the labels_ array and transit_matrix_. 

    transit_matrix_ : matrix
        Corresponds to the Markov transition matrix learned during the fit process.

    Examples
    -------- 
    >>> from dsbox.ml.markov import MarkovSequenceMiner

    >>> x_train = ['a', 'a', 'b', 'c', 'd', 'a']
    >>> clf = MarkovSequenceMiner()
    >>> clf.fit(x_train)
    MarkovSequenceMiner(equiproba=False, init_label='a', warm_start=False)

    >>> x_test = ['a', 'b', 'c', 'd', 'e', 'a']
    >>> chained_p, transition_p = clf.predict_proba(x_test)
    >>> print(chained_p)
    [1.  0.5 0.5 0.5 0.  0. ]
    >>> print(transition_p)
    [1.  0.5 1.  1.  0.  0. ]

    """

    def __init__(self, init_label=None, warm_start=False, equiproba=False):
        self.init_label = init_label
        self.equiproba = equiproba
        self.warm_start = warm_start
        self.init_vector = None
        self.cur_vector = None

        self.attr_to_check = ["labels_",
                              "inv_dict_labels_",
                              "transit_matrix_"
                              ]

    def fit(self, X, y=None):
        """
        Fit the model by estimating probabilities between two states.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str
        y : array-like, shape (n_samples,)
            No used, only here for compatibility reason

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        x = column_or_1d(X)

        if self.init_label is None:
            self.init_label = x[0]

        # get all different labels of the sequence
        # keep their index in an reverse dict and build the transition matrix
        self.labels_ = np.unique(x)
        label_amount = len(self.labels_)
        self.inv_dict_labels_ = {self.labels_[i]: i for i in range(0, label_amount)}
        self.transit_matrix_ = np.zeros([label_amount, label_amount])

        # initialize vector with the first event
        self.init_vector = np.zeros(label_amount)
        self.init_vector[self.inv_dict_labels_[self.init_label]] = 1

        if self.equiproba:
            self.init_vector = np.ones(label_amount) * (1.0 / label_amount)

        for i in range(1, len(x)):
            previous_event_index = self.inv_dict_labels_[x[i - 1]]
            current_event_index = self.inv_dict_labels_[x[i]]
            self.transit_matrix_[previous_event_index, current_event_index] += 1

        for i in range(0, self.transit_matrix_.shape[0]):
            self.transit_matrix_[i, :] = self.transit_matrix_[i, :] / self.transit_matrix_[i, :].sum()

        return self

    def predict_proba(self, X):
        """
        Return two arrays of probabilities for each state transition in sequence X.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        chained_p_ : array-like
            Chained probabilities
        transition_p_ : array-like
            Transition probabilities
        """
        check_is_fitted(self, self.attr_to_check)
        x = column_or_1d(X)

        # initialize the current vector that evolves with each new event
        if not self.warm_start or self.cur_vector is None:
            self.compute_probas_ = np.zeros([len(x), len(self.labels_)])
            self.compute_probas_[0, :] = self.init_vector
            self.cur_vector = np.zeros(len(self.labels_))
            self.cur_vector[self.inv_dict_labels_[x[0]]] = 1
            self.cur_vector = self.cur_vector.reshape(1, len(self.cur_vector))
            self.chained_p_ = np.ones(len(x))
            self.transition_p_ = np.ones(len(x))

        for i in range(1, len(x)):
            self.cur_vector = np.dot(self.cur_vector, self.transit_matrix_)
            self.compute_probas_[i, :] = self.cur_vector[0, :]

            if x[i - 1] not in self.inv_dict_labels_ or x[i] not in self.inv_dict_labels_:
                self.transition_p_[i] = 0.0
                self.chained_p_[i] = 0.0
            else:
                previous_event_index = self.inv_dict_labels_[x[i - 1]]
                current_event_index = self.inv_dict_labels_[x[i]]
                self.transition_p_[i] = self.transit_matrix_[previous_event_index, current_event_index]
                self.chained_p_[i] = self.cur_vector[0, current_event_index]

        return self.chained_p_, self.transition_p_

    def predict(self, X):
        """
        For each transition in sequence X, returns a boolean value, which is True if a transition was the most
         probable transition to occur, and False if not.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        bool_vector : array-like
            Collection of boolean values

        """
        self.predict_proba(X)
        x = column_or_1d(X)
        bool_vector = np.zeros(len(x), dtype=bool)

        for i in range(0, len(x)):
            proba_vector = self.compute_probas_[i]
            max_proba = proba_vector.max()
            i_max_labels = set()
            while max_proba == proba_vector.max():
                i_max = proba_vector.argmax()
                i_max_labels.add(self.labels_[i_max])
                proba_vector[i_max] = -1

            if x[i] in i_max_labels:
                bool_vector[i] = True

        return bool_vector

    def predict_next(self, X):
        """
        For a given sequence X, returns the most probable event(s) to occur.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        predicted_next_event : array-like
            Most probable next event to occur

        """
        self.predict_proba(X)
        proba_vector = self.cur_vector
        proba_vector = np.dot(proba_vector, self.transit_matrix_)
        max_proba = proba_vector.max()
        predicted_next_event = []

        while max_proba == proba_vector.max():
            i_max = proba_vector.argmax()
            predicted_next_event.append(self.labels_[i_max])
            proba_vector[0, i_max] = -1

        return np.array(predicted_next_event)


class TimeDurationSequenceMiner(BaseEstimator):
    """ Sequence time distribution miner.
    
    The purpose of this class is to associate Symbolic data analysis (SDA) to state transitions in a sequence. In 
    this particular case, the time duration distribution is computed and kept for each state transition. It  
    gives information about the normal (or not) behavior on transition, using the time duration distribution point
    of view.
    
    Parameters
    ----------
    bins : int, optional (default=3)
        Set the number of bins to use to estimate the "discrete" distribution using an histogram
        
    
    Attributes
    ----------
    predict_threshold : float, (default=0.5)
        Used by the predict method, to decide if the probablity outputs a True value (strictly above the threshold)
        or a False value (equal or below the threshold)
        
    labels_ : array of all the states
        Corresponds to states seen during the fit process.
    
    inv_dict_labels_ : dict
        For a given state, returns its position in the labels_ array and transit_matrix_.
        
    timeduration_matrix_ : matrix
        Corresponds to the time durations between states (each cell contains a list of values).
        
    sda_matrix_ : matrix
        Corresponds to the time duration distribution between states (each celle contains an HistogramData object).
    
    References
    ----------
    .. [1] Lynne Billard, Edwin Diday. Symbolic Data Analysis: Conceptual Statistics and Data Mining. 
    John Wiley & Sons. ISBN 978-0-470-09017-6, 2012
    
    Examples
    --------
    >>> import pandas as pd
    >>> from dsbox.ml.markov import TimeDurationSequenceMiner
    
    >>> df = pd.DataFrame({'event': ['a', 'a', 'b', 'b', 'a', 'a', 'a', 'b'], \
    'timestamp': [0, 12, 14, 29, 40, 42, 44, 48]})
                           
    >>> tds = TimeDurationSequenceMiner()
    >>> tds.fit(df[['event', 'timestamp']].values)
    TimeDurationSequenceMiner(bins=3)
    
    >>> df_test = pd.DataFrame({'event': ['a', 'b', 'b', 'a', 'b', 'a'], \
                                'timestamp': [0, 2, 20, 24, 27, 38]})
                                
    >>> y_test = tds.predict_proba(df_test[['event', 'timestamp']].values)
    >>> print(y_test)
    [1.  0.5 0.  0.  0.  1. ]
    
    
    """

    def __init__(self, bins=3):
        self.bins = bins
        self.predict_threshold = 0.5

        self.attr_to_check = ["labels_",
                              "inv_dict_labels_",
                              "timeduration_matrix_",
                              "sda_matrix_"
                              ]

    def fit(self, X, y=None, log_base=False):
        """
        Fit the model by estimating time duration distribution between states. 

        Parameters
        ----------
        X : array-like, shape = (n_samples, 2)
            First column corresponds to states, second column to timestamps
        y : array-like, shape (n_samples,)
            No used, only here for compatibility reason

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        if X.shape[1] != 2:
            raise ValueError("Shape must be exactly (n,2) but is " + X.shape)

        x = X[:, 0]
        timestamp = X[:, 1]

        x = column_or_1d(x)
        timestamp = column_or_1d(timestamp)

        self.labels_ = np.unique(x)
        label_amount = len(self.labels_)
        self.inv_dict_labels_ = {self.labels_[i]: i for i in range(0, label_amount)}
        self.sda_matrix_ = np.empty([label_amount, label_amount], dtype=object)
        self.timeduration_matrix_ = np.empty([label_amount, label_amount], dtype=object)
        for i in range(0, label_amount):
            for j in range(0, label_amount):
                self.timeduration_matrix_[i, j] = list()

        for i in range(1, len(x)):
            time_duration = timestamp[i] - timestamp[i - 1]
            previous_event_index = self.inv_dict_labels_[x[i - 1]]
            current_event_index = self.inv_dict_labels_[x[i]]
            self.timeduration_matrix_[previous_event_index, current_event_index].append(time_duration)

        for i in range(0, label_amount):
            for j in range(0, label_amount):
                if self.timeduration_matrix_[i, j] is not None:
                    if log_base:
                        hist = np.histogram(np.log(self.timeduration_matrix_[i, j]), bins=self.bins)
                    else:
                        hist = np.histogram(self.timeduration_matrix_[i, j], bins=self.bins)

                    hda = HistogramData(hist[1], hist[0])
                    self.sda_matrix_[i, j] = hda

        return self

    def predict_proba(self, X):
        """
        Return an array corresponding to probabilities for having these time durations
        between states of X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, 2)
            First column corresponds to states, second column to timestamps

        Returns
        -------
        time_p_: array-like
            Time duration probabilities
        """

        if X.shape[1] != 2:
            raise ValueError("Shape must be exactly (n,2) but is " + X.shape)

        x = X[:, 0]
        timestamp = X[:, 1]

        check_is_fitted(self, self.attr_to_check)

        x = column_or_1d(x)
        timestamp = column_or_1d(timestamp)

        self.time_p_ = np.ones(len(x))

        for i in range(1, len(x)):
            if x[i - 1] in self.inv_dict_labels_ and x[i] in self.inv_dict_labels_:
                time_duration = timestamp[i] - timestamp[i - 1]
                previous_event_index = self.inv_dict_labels_[x[i - 1]]
                current_event_index = self.inv_dict_labels_[x[i]]

                self.time_p_[i] = self.sda_matrix_[previous_event_index, current_event_index].get_amount(time_duration,
                                                                                                         normalized=True)

        return self.time_p_

    def predict(self, X):
        """
        For each transition in sequence X, returns a boolean value, which is True if the time
        duration probability between 2 states is more than the threshold (default: 0.5) and False if not.

        Parameters
        ----------
        X : array-like, shape = (n_samples, 2)
            First column corresponds to states, second column to timestamps

        Returns
        -------
        bool_vector : array-like
            Collection of boolean values

        """
        return self.predict_proba(X) > self.predict_threshold


class MetaSequenceMiner(BaseEstimator):
    """ Meta-model for sequence mining.
    
    This class uses MarkovSequenceMiner (transition probabilities between states) and TimeDurationSequenceMiner 
    (time duration distributions between states) within a given event window to predict states probability occurence.
        
    Attributes
    ----------
    predict_threshold : float, (default=0.5)
        Used by the predict method, to decide if the probablity outputs a True value (strictly above the threshold)
        or a False value (equal or below the threshold)
    
    markovseqminer : MarkovSequenceMiner
        Used to compute the transition probabilities between states.
        
    timeseqminer : TimeDurationSequenceMiner
        Used to compute the time duration distributions between states
    
    Examples
    --------
    >>> import pandas as pd
    >>> from dsbox.ml.markov import MetaSequenceMiner
    
    >>> df = pd.DataFrame({'event': ['a', 'a', 'b', 'b', 'a', 'a', 'a', 'b'], \
                           'timestamp': [0, 12, 14, 29, 40, 42, 44, 48] })
    >>> metaseqminer = MetaSequenceMiner()
    >>> metaseqminer.fit(df[['event', 'timestamp']].as_matrix())
    MetaSequenceMiner(bins=10)
    
    >>> df_test = pd.DataFrame({'event': ['a', 'b', 'b', 'a', 'b', 'a'], \
                                'timestamp': [0, 2, 20, 24, 27, 38]})
    >>> metaseqminer.predict_proba(df_test[['event', 'timestamp']].as_matrix()).tolist()
    [1.0, 0.45, 0.22, 0.278, 0.22220000000000004, 0.77778]
    
    """

    def __init__(self, bins=10):
        self.bins = bins
        self.markovseqminer = MarkovSequenceMiner()
        self.timeseqminer = TimeDurationSequenceMiner(bins=self.bins)
        self.predict_threshold = 0.5

    def fit(self, X, y=None):
        """
        Fit the meta-model by estimating probabilities transition and time duration distribution between states. 

        Parameters
        ----------
        X : array-like, shape = (n_samples, 2)
            First column corresponds to states, second column to timestamps
        y : array-like, shape (n_samples,)
            No used, only here for compatibility reason

        Returns
        -------
        self : object
            Returns an instance of self.
        """

        if X.shape[1] != 2:
            raise ValueError("Shape must be exactly (n,2) but is " + X.shape)

        self.markovseqminer.fit(X[:, 0])
        self.timeseqminer.fit(X)

        return self

    def predict_proba(self, X, operation=np.mean):
        """
        Return an array corresponding to probabilities for having these transitions and time durations
        between states of X.

        Parameters
        ----------
        X : array-like, shape = (n_samples, 2)
            First column corresponds to states, second column to timestamps
            
        operation : callable, optional (default: numpy.mean)
            Set the function used to merge transitions and time durations probabilities.
        
        Returns
        -------
        time_p_: array-like
            Mean between transition probabilities and time duration probabilities
        """

        if X.shape[1] != 2:
            raise ValueError("Shape must be exactly (n,2) but is " + X.shape)

        markov_probas, _ = self.markovseqminer.predict_proba(X[:, 0])
        sda_probas = self.timeseqminer.predict_proba(X)

        df_probas = pd.DataFrame({'markov_p': markov_probas,
                                  'sda_p': sda_probas})

        df_probas['score'] = df_probas.apply(operation, axis=1)

        return df_probas['score'].values

    def predict(self, X):
        """
        For each transition in sequence X, returns a boolean value, which is True if predict_proba result
         is more than the threshold (default: 0.5) and False if not.

        Parameters
        ----------
        X : array-like, shape = (n_samples, 2)
            First column corresponds to states, second column to timestamps

        Returns
        -------
        bool_vector : array-like
            Collection of boolean values

        """
        return self.predict_proba(X) > self.predict_threshold
