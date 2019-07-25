import unittest

import numpy as np
import pandas as pd

from dsbox.ml.markov import MarkovSequenceMiner, TimeDurationSequenceMiner, MetaSequenceMiner


class TestMarkovSequenceMiner(unittest.TestCase):
    def test_chained_probas_with_transition_proba_matrix_prefilled(self):
        # given
        pmatrix = np.array([[0.5, 0.25, 0.25],
                             [0.5, 0, 0.5],
                             [0.25, 0.25, 0.5]])

        # when
        clf = MarkovSequenceMiner(equiproba=True)
        clf.transit_matrix_ = pmatrix

        clf.labels_ = np.array(['Rain', 'Nice', 'Snow'])
        clf.inv_dict_labels_ = {clf.labels_[i]: i for i in range(0, len(clf.labels_))}
        clf.init_vector = np.ones(len(clf.labels_)) * (1.0 / len(clf.labels_))

        clf.predict_proba(['Nice', 'Nice', 'Rain', 'Rain'])
        proba_vector = clf.cur_vector.tolist()[0]

        # then
        values_to_obtain = np.array([0.406, 0.188, 0.406])

        self.assertEqual(np.round(proba_vector[0], 3), values_to_obtain[0])
        self.assertEqual(np.round(proba_vector[1], 3), values_to_obtain[1])
        self.assertEqual(np.round(proba_vector[2], 3), values_to_obtain[2])

    def test_fit_predict_proba_simple_markov_chain_should_return_correct_values(self):
        # given
        sequence = ['a', 'a', 'b', 'c', 'd', 'a']

        # when
        clf = MarkovSequenceMiner()
        clf.fit(sequence)
        clf.predict_proba(['a', 'b', 'c', 'd', 'e', 'a'])
        y = clf.cur_vector.tolist()[0]

        # then
        y_true = [0.531, 0.281, 0.062, 0.125]
        self.assertEqual(np.round(y[0], 3), y_true[0])
        self.assertEqual(np.round(y[1], 3), y_true[1])
        self.assertEqual(np.round(y[2], 3), y_true[2])

    def test_fit_predict_proba_simple_markov_chain_with_warm_start_ie_keep_fitted_state(self):
        # given
        sequence = ['a', 'a', 'b', 'c', 'd', 'a']

        # when
        clf = MarkovSequenceMiner(warm_start=True)
        clf.fit(sequence)
        clf.predict_proba(['a', 'b', 'c', 'd', 'e'])
        clf.predict_proba(['e', 'a'])
        y = clf.cur_vector.tolist()[0]

        # then
        y_true = [0.531, 0.281, 0.062, 0.125]
        self.assertEqual(np.round(y[0], 3), y_true[0])
        self.assertEqual(np.round(y[1], 3), y_true[1])
        self.assertEqual(np.round(y[2], 3), y_true[2])

    def test_chained_transition_probas_which_are_computed_during_fitting(self):
        # given
        sequence = ['a', 'a', 'b', 'c', 'd', 'a']

        # when
        clf = MarkovSequenceMiner()
        clf.fit(sequence)
        chained_p, transition_p = clf.predict_proba(['a', 'b', 'c', 'd', 'e', 'a'])

        # then
        chained_p_true = [1.0, 0.5, 0.5, 0.5, 0.0, 0.0]
        transition_p_true = [1.0, 0.5, 1.0, 1.0, 0.0, 0.0]

        self.assertListEqual(chained_p.tolist(), chained_p_true)
        self.assertListEqual(transition_p.tolist(), transition_p_true)

    def test_fit_predict_which_should_return_correct_anomalies(self):
        # given
        sequence = ['a', 'a', 'b', 'c', 'a', 'd', 'd', 'd', 'a']

        # when
        clf = MarkovSequenceMiner()
        clf.fit(sequence)
        y = clf.predict(['c', 'b', 'd', 'd', 'a']).tolist()

        # then
        y_true = [False, False, True, True, True]
        self.assertEqual(y, y_true)

    def test_fit_predict_next_which_should_return_next_most_likely(self):
        # given
        sequence = ['a', 'a', 'b', 'c', 'a', 'd', 'd', 'd', 'a']

        # when
        clf = MarkovSequenceMiner()
        clf.fit(sequence)
        y = clf.predict_next(['c', 'b', 'd', 'd', 'a'])

        # then
        y_true = ['a']
        self.assertEqual(y, y_true)


class TestTimeDurationSequenceMiner(unittest.TestCase):
    def test_fit_predict_probas_should_return_correct_values(self):
        # given
        df = pd.DataFrame({'event': ['a', 'a', 'b', 'b', 'a', 'a', 'a', 'b'],
                           'timestamp': [0, 12, 14, 29, 40, 42, 44, 48]
                           })

        # when
        tds = TimeDurationSequenceMiner()
        tds.fit(df[['event', 'timestamp']].values)

        df_test = pd.DataFrame({'event': ['a', 'b', 'b', 'a', 'b', 'a'],
                                'timestamp': [0, 2, 20, 24, 27, 38]
                                })

        y_test = tds.predict_proba(df_test[['event', 'timestamp']].values)

        # then
        y_true = [1, 0.5, 0, 0, 0, 1]
        self.assertListEqual(y_true, y_test.tolist())

    def test_fit_predict_should_return_correct_anomalies(self):
        # given
        df = pd.DataFrame({'event': ['a', 'a', 'b', 'b', 'a', 'a', 'a', 'b'],
                           'timestamp': [0, 12, 14, 29, 40, 42, 44, 48]
                           })

        # when
        tds = TimeDurationSequenceMiner()
        tds.fit(df[['event', 'timestamp']].values)

        df_test = pd.DataFrame({'event': ['a', 'b', 'b', 'a', 'b', 'a'],
                                'timestamp': [0, 2, 20, 24, 27, 38]
                                })

        y_test = tds.predict(df_test[['event', 'timestamp']].values)

        # then
        y_true = [True, False, False, False, False, True]
        self.assertListEqual(y_true, y_test.tolist())


class TestMetaSequenceMiner(unittest.TestCase):
    def test_metasequenceminer_fit_predict_proba_should_return_correct_values(self):
        # given
        df = pd.DataFrame({'event': ['a', 'a', 'b', 'b', 'a', 'a', 'a', 'b'],
                           'timestamp': [0, 12, 14, 29, 40, 42, 44, 48]
                           })

        # when
        metaseqminer = MetaSequenceMiner()
        metaseqminer.fit(df[['event', 'timestamp']].values)

        df_test = pd.DataFrame({'event': ['a', 'b', 'b', 'a', 'b', 'a'],
                                'timestamp': [0, 2, 20, 24, 27, 38]
                                })

        y_test = metaseqminer.predict_proba(df_test[['event', 'timestamp']].values)

        # then
        y_true = [1.0, 0.45, 0.22, 0.28, 0.22, 0.78]
        self.assertListEqual(y_true, y_test.round(2).tolist())
