import unittest

import pandas as pd
from dsbox.ml.text import LSH
from pandas.util.testing import assert_frame_equal


class TestLSH(unittest.TestCase):
    def test_get_hash(self):
        # given
        clf = LSH(n_hash_functions=3, n_bands=3)
        clf._seeds = [1, 2, 3]

        # when
        hashes = []
        for i in range(0, len(clf._seeds)):
            hashes.append(clf._get_hash(i, 'toto'))

        # then
        hashes_expected = [10143814839887274982, 8018333638267422442, 9635205340518008203]
        self.assertListEqual(hashes_expected, hashes)

    def test_minhash(self):
        # given
        clf = LSH(n_hash_functions=3, n_bands=3)
        clf._seeds = [1, 2, 3]

        # when
        # string 'octo technology' manually shinglized
        shingles = ['oct', 'cto', 'to ', 'o t', ' te', 'tec', 'ech', 'chn', 'hno', 'nol', 'olo', 'log', 'ogy']
        minhashes = list(clf._minhash(shingles))

        # then
        minhashes_expected = [3.8649734935051917e+17, 2.1739818134115853e+17, 1.4207452988806428e+18]
        self.assertListEqual(minhashes_expected, minhashes)

    def test_build_hashes(self):
        # given
        clf = LSH(n_hash_functions=3, n_bands=3)
        clf._seeds = [1, 2, 3]

        # when
        texts = ['octo', 'technology']
        hashes = clf._build_hashes(texts, 3)

        # then
        self.assertListEqual([6.6663064401848566e+18, 2.1739818134115853e+17, 4.2542837112525629e+18], list(hashes[0]))
        self.assertListEqual([3.8649734935051917e+17, 1.2233874801797798e+18, 1.4207452988806428e+18], list(hashes[1]))

    def test_reverse_hash(self):
        # given
        clf = LSH(n_hash_functions=3, n_bands=3)
        clf._seeds = [1, 2, 3]  # fixing seeds for test

        # when
        # 2 string 'octo', 'technology' manually shinglized
        minhashes = []
        shingles = ['oct', 'cto']
        minhashes.append(list(clf._minhash(shingles)))
        shingles = ['tec', 'ech', 'chn', 'hno', 'nol', 'olo', 'log', 'ogy']
        minhashes.append(list(clf._minhash(shingles)))

        reverse_hashes = clf._reverse_hash(minhashes, 3, 1)

        # then
        reverse_hashes_expected = [
            {6.6663064401848566e+18: set([0]), 3.8649734935051917e+17: set([1])},
            {2.1739818134115853e+17: set([0]), 1.2233874801797798e+18: set([1])},
            {4.2542837112525629e+18: set([0]), 1.4207452988806428e+18: set([1])}]

        self.assertListEqual(reverse_hashes_expected, reverse_hashes)

    def test_candidates(self):
        # given
        texts = ["Luc, la force en toi tu dois trouver",
                 "Trouver la force en toi tu dois Luc",
                 "Luc, Tu dois trouver la force en toi",
                 "Trouver tu dois la force en toi Luc",
                 "Tu dois la force trouver en toi",

                 "Un anneau pour les gouverner tous",
                 "Un anneau pour tous les gouverner",
                 "Tous les gouverner par le pouvoir de l'anneau",
                 "L'anneau unique pour dans les ténèbres les lier",
                 ]

        # when
        clf = LSH(n_hash_functions=99, n_bands=33)
        clf._seeds = list(range(0, 99))  # fixing seeds for test
        clf.fit(texts)

        texts_to_compare = ["La force je dois trouver en moi",
                            "L'anneau de pouvoir pour tous les gouverner"
                            ]

        candidates = clf._get_candidates(texts_to_compare)

        # then

        candidates_expected = [[4, 1, 2, 3], [7, 6, 4, 1, 2, 3, 5]]
        self.assertSetEqual(set(candidates_expected[0]), set(candidates[0]))
        self.assertSetEqual(set(candidates_expected[1]), set(candidates[1]))

        # just for displaying test results
        """for i in range(0, len(texts_to_compare)):
            print(texts_to_compare[i])
            print('--------')
            for k in candidates[i]:
                print(texts[k])

            print()"""

    def test_predict(self):
        # given
        texts = ["trouver la force en toi tu dois luc",
                 "Luc, Tu dois trouver la force en toi",
                 "Trouver tu dois la force en toi Luc",
                 "Tu dois la force trouver en toi",

                 "Un anneau pour les gouverner tous",
                 "Un anneau pour tous les gouverner",
                 "Tous les gouverner par le pouvoir de l'anneau",
                 "L'anneau unique pour dans les ténèbres les lier",
                 ]

        # when
        clf = LSH(n_hash_functions=200, n_bands=50)
        clf._seeds = list(range(0, 200))  # fixing seeds for test
        clf.fit(texts)

        texts_to_compare = ["La force je dois trouver en moi",
                            "L'anneau de pouvoir pour tous les gouverner"
                            ]

        candidates = clf.predict_proba(texts_to_compare)

        # then

        candidates_expected = pd.DataFrame([
            ['Luc, Tu dois trouver la force en toi', 0.717948717948718],
            ["Tous les gouverner par le pouvoir de l'anneau", 0.7959183673469388]])

        assert_frame_equal(candidates_expected, candidates)
