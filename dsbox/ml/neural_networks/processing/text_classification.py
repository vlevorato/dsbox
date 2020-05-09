import numpy as np
from gensim import corpora
from keras.preprocessing import sequence
from nltk.tokenize import TreebankWordTokenizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted, column_or_1d

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class Text2Sequence(BaseEstimator, TransformerMixin):
    """
    Class used to transform text into integers sequence.
    
    Parameters
    ----------
    pad_string : string, optional (default='')
        string value used for padding sequence.
        
    seq_len : int, optional, (default=None)
        fixed sequence length used to build output sequence. If None, this length will be computed during
        transform operation, as the mean of all original sequences + 2 times standard deviation.
        
    tokenizer : object, optional (default=TreebankWordTokenizer())
        object used to tokenize text. The class used must implements a tokenize method.
        
    stemmer : object, optional (default=None)
        object used to stem text. The class used must implements a stem method.
   
   
    Attributes
    ----------
    dictionary_ : gensim.corpora.Dictionary
        maps all tokens to an unique id.
        
    dictionary_size_ : int
        size of the dictionary_ attribute.
            
    Examples
    --------
    >>> import numpy as np
    >>> from nltk.stem.snowball import EnglishStemmer
    >>> from dsbox.ml.neural_networks.processing import Text2Sequence
    >>> X = np.array(['this is really awesome !', \
                      'this is really crap !!'])

    >>> text2seq = Text2Sequence(stemmer=EnglishStemmer())
    >>> text2seq.fit_transform(X)
    array([[0, 5, 3, 4, 2, 1],
           [5, 3, 4, 6, 1, 1]], dtype=int32)
    """

    def __init__(self, pad_string='', tokenizer=TreebankWordTokenizer(), stemmer=None):

        self.pad_string = pad_string
        self.seq_len = None
        self.tokenizer = tokenizer
        self.stemmer = stemmer

        self.attr_to_check = ["dictionary_",
                              "dictionary_size_",
                              ]

    def tokenize(self, text_list):
        """
        Takes a list of texts and return a list of tokens for each text (stemmed or not).
        
        Parameters
        ----------
        text_list : array-like, iterable
            list of strings

        Returns
        -------
            list of token lists

        """
        all_tokens = []
        for text in text_list:
            tokens = self.tokenizer.tokenize(text)
            if self.stemmer is not None:
                tokens = [self.stemmer.stem(word) for word in tokens]
            all_tokens.append(tokens)
        return all_tokens

    def fit(self, X, y=None):
        """
        Fit the transformer by building the word corpus.

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
        all_tokens = self.tokenize(x)
        # adding blank line for considering padding value
        self.dictionary_ = corpora.Dictionary([[self.pad_string]] + all_tokens)

        self.dictionary_size_ = len(self.dictionary_.keys())

        return self

    def transform(self, X):
        """
        Transform a list of texts into sequences of integers (ids), based on a known corpus. If a word is not
        found in the corpus built during fit operation, it will be replaced by the padding string value.
        
        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str
        
        Returns
        -------
        numpy array of ids sequences.
        
        """

        check_is_fitted(self, self.attr_to_check)
        x = column_or_1d(X)

        new_tokens = self.tokenize(x)
        pad_value = self.dictionary_.token2id[self.pad_string]

        # transform all tokens into unique ids
        word_ids, word_ids_len = [], []
        for doc in new_tokens:
            word_id = []
            for word in doc:
                if word in self.dictionary_.token2id:
                    word_id.append(self.dictionary_.token2id[word])
                else:
                    word_id.append(pad_value)

            word_ids.append(word_id)
            word_ids_len.append(len(word_id))

        # compute the length sequence
        if self.seq_len is None:
            self.seq_len = np.round((np.mean(word_ids_len) + 2 * np.std(word_ids_len))).astype(int)

        return sequence.pad_sequences(np.array(word_ids), maxlen=self.seq_len, value=pad_value)
