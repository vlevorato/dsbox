import random
import sys
import xxhash
from collections import Counter, OrderedDict

import numpy as np
import pandas as pd
from dsbox.ml.metrics import generalized_jaccard_similarity_score
from dsbox.ml.text.utils import shinglelize
from sklearn.base import BaseEstimator
from sklearn.utils.validation import check_array, check_is_fitted

__author__ = "Vincent Levorato"
__credits__ = "https://github.com/octo-technology/bdacore"
__license__ = "Apache 2.0"


class LSH(BaseEstimator):
    """ Local Sensitive Hashing classifier for texts.

    Comparison of texts can need heavily computing if done with brute force, as string comparison is 
    costly. The problem of finding textually similar documents can be turned into such a set problem 
    by the technique known as shingling. Then, using a technique called minhashing, the classifier
    compresses large sets in such a way that we can still deduce the similarity of the underlying sets
    from their compressed versions, approximating Jaccard similarity coefficient.
    Another important problem that arises when we search for similar items of any kind is that there 
    may be far too many pairs of items to test each pair for their degree of similarity, even if 
    computing the similarity of any one pair can be made very easy. That concern motivates a technique 
    called locality-sensitive hashing, for focusing our search on pairs that are most likely to be similar.


            Parameters
            ----------
            n_hash_functions: int, optional, default 100
                amount of hash functions to use

            shingle_size: int, optional, default 3
                size of shingles

            most_common: int, optional, default 10
                amount of candidates to return during the candidates retrieval phase (hash collisions)

            n_bands: int, optional, default 50
                amount of bands to use for merging hash codes. Must be a multiple of the amount of hash functions

            compare_function: callable, optional, default bdacore.metrics.generalized_jaccard_similarity_score
                metric used to sort found candidates

            Attributes
            ----------
            string_minhashes_ : array
                list of minhashes for each hash function for each shinglelized text

            band_size_ : int
                size of one band

            hash_neighbors_bands_ : list
                buckets of hashes merged into bands

            candidates_ : list of tuples
                full results of predict method with all candidates per text with associated scores

            References
            ----------
            .. [1] Jure Leskovec, Anand Rajaraman, Jeff Ullman. Mining of Massive Datasets, Chapter 3:
            Finding Similar Items, Cambridge University Press, 2011

            Examples
            --------  
            >>> from dsbox.ml.text import LSH
            >>> import random
            >>> random.seed(42) # for making doctests stable

            >>> texts = ['there is a better way', 'demain, avant après-demain, après hier']
            >>> texts_to_compare = ['hier, dès demain']
            >>> clf = LSH()
            >>> clf.fit(texts)
            >>> clf.predict_proba(texts_to_compare)
                                                    0         1
            0  demain, avant après-demain, après hier  0.421053


    """

    def __init__(self, n_hash_functions=100, shingle_size=3,
                 most_common=10, n_bands=50, compare_function=generalized_jaccard_similarity_score):
        try:
            if n_hash_functions % n_bands > 0:
                raise ValueError("Amount of bands must be a multiple of the amount of hash functions")
        except:
            print(str(n_hash_functions) + " divide by " + str(n_bands) + " gives a float value, not int.")

        self._seeds = []
        self._shingle_size = shingle_size
        self.n_bands = n_bands
        self.most_common = most_common
        self.compare_function = compare_function

        self._hashfunctions = []
        self.n_hash_functions = n_hash_functions
        self._init_hash_functions(n_hash_functions)

        self.attr_to_check = ["string_minhashes_",
                              "band_size_",
                              "hash_neighbors_bands_",
                              "X_"]

    def _init_hash_functions(self, n):
        for i in range(0, n):
            self._seeds.append(random.randint(1, 1000000))

    def _get_hash(self, index_function, value):
        x = xxhash.xxh64(value, seed=self._seeds[index_function])
        x = x.intdigest()

        return x

    def _minhash(self, shingles):
        hashes = np.zeros(self.n_hash_functions)

        for i in range(0, self.n_hash_functions):
            min_hash = sys.maxsize
            for j in range(0, len(shingles)):
                val_hash = self._get_hash(i, shingles[j])
                if val_hash < min_hash:
                    min_hash = val_hash
            hashes[i] = min_hash

        return hashes

    def _build_hashes(self, string_vector, shingle_size):
        string_minhashes = np.array(np.zeros(len(string_vector)), dtype=object)
        for i in range(0, len(string_vector)):
            string_minhashes[i] = self._minhash(shinglelize(string_vector[i], shingle_size))

        return string_minhashes

    def _reverse_hash(self, minhashes, n_bands, band_size):

        hash_neighbors_bands = []

        for band in range(0, n_bands):

            hash_dict = {}

            for i in range(0, len(minhashes)):

                hash_code_band = 0
                for h in range(int(band) * int(band_size), (int(band) + 1) * int(band_size)):
                    hash_code_band += minhashes[i][h]

                if hash_code_band not in hash_dict:
                    s = set()
                    s.add(i)
                    hash_dict[hash_code_band] = s
                else:
                    hash_dict[hash_code_band].add(i)

            hash_neighbors_bands.append(hash_dict)

        return hash_neighbors_bands

    def fit(self, X, y=None):
        """ Fit the model by shingling and hashing strings.

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

        check_array(X, ensure_2d=False, dtype='str')

        self.X_ = X

        if self.n_bands is None:
            self.n_bands = self.n_hash_functions

        self.band_size_ = self.n_hash_functions / self.n_bands

        self.string_minhashes_ = self._build_hashes(X, self._shingle_size)

        self.hash_neighbors_bands_ = self._reverse_hash(self.string_minhashes_, self.n_bands, self.band_size_)

    def _get_candidates(self, raw_documents):
        """ For each given string, return a list of the closest strings known by the estimator. 
        Return results are ordered by the amount of "collisions" with the hash keys of the given string each
        known string made.

        Parameters
        ----------
        raw_documents : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        C : array-like, iterable
            Collection of str
        """

        string_minhashes_to_predict = self._build_hashes(raw_documents, self._shingle_size)
        hash_neighbors_bands_to_predict = self._reverse_hash(string_minhashes_to_predict, self.n_bands, self.band_size_)

        documents_candidates = []
        for i in range(0, len(raw_documents)):
            documents_candidates.append(list())

        for band in range(0, len(hash_neighbors_bands_to_predict)):
            candidates = []
            for hash_code_band_to_predict, ids_doc in hash_neighbors_bands_to_predict[band].items():
                if hash_code_band_to_predict in self.hash_neighbors_bands_[band]:
                    candidates = candidates + list(self.hash_neighbors_bands_[band].get(hash_code_band_to_predict))

                for id in ids_doc:
                    documents_candidates[id] = documents_candidates[id] + candidates

        for i in range(0, len(documents_candidates)):
            documents_candidates[i] = Counter(documents_candidates[i]).most_common(self.most_common)
            documents_candidates[i].sort(key=lambda tup: tup[1], reverse=True)
            candidates = list()
            for k in documents_candidates[i]:
                candidates.append(k[0])
            documents_candidates[i] = candidates

        return documents_candidates

    def predict_proba(self, X):
        """Return the closest learned candidate for each text given in input, associated to metric score
        passed in self.compare_function.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        df_candidates : dataframe
            Text candidates with metric score.
        """

        check_is_fitted(self, self.attr_to_check)

        candidates_indexes = self._get_candidates(X)
        candidates = []
        for i in range(0, len(X)):
            candidates.append(dict())

        for i in range(0, len(X)):
            candidate_index = candidates_indexes[i]
            for j in candidate_index:
                candidate = self.X_[j]
                candidates[i][candidate] = self.compare_function(X[i], candidate)

        df_list = []

        for i in range(0, len(X)):
            candidates[i] = list(OrderedDict(sorted(candidates[i].items(), key=lambda t: t[1], reverse=True)).items())
            df_list.append(pd.DataFrame([candidates[i][0]]))

        self.candidates_ = candidates

        df_candidates = pd.concat(df_list, ignore_index=True)

        return df_candidates

    def predict(self, X):
        """Return the closest learned candidate for each text given in input.

        Parameters
        ----------
        X : array-like, iterable
            Collection of str or an iterable which yields str

        Returns
        -------
        df_candidates : array-like, shape (n_samples,)
            Text candidates.
        """

        df_candidates = self.predict_proba(X)
        return df_candidates[0]
