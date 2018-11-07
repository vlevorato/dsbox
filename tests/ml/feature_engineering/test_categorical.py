import unittest

import pandas as pd
from pandas.util.testing import assert_series_equal, assert_frame_equal

from dsbox.ml.feature_engineering import project_continuous_on_categorical, CategoricalProjector, TagEncoder


class TestCategorical(unittest.TestCase):
    def test_project_continuous_on_categorical_function_should_return_correct_results(self):
        # given
        df = pd.DataFrame({'item': ['A', 'A', 'B', 'B', 'B'],
                           'price': [0.0, 1.0, 2.0, 2.0, 1.0]})

        # when
        serie_transformed = project_continuous_on_categorical(df, 'item', 'price')

        # then
        serie_expected = pd.Series(data=[0.5, 0.5, 1.66666667, 1.66666667, 1.66666667],
                                   index=df.index,
                                   name='price')

        assert_series_equal(serie_expected, serie_transformed)

    def test_categoricalprojector_should_project_continuous_on_categorical_data_properly(self):
        # given
        df = pd.DataFrame({'item': ['A', 'A', 'B', 'B', 'B'],
                           'town': ['Paris', 'London', 'Paris', 'Paris', 'Roma'],
                           'price': [0.0, 1.0, 2.0, 2.0, 1.0],
                           'quantity': [3, 4, 1, 2, 1]
                           })

        # when
        cat_projector = CategoricalProjector(['item', 'town'], ['price', 'quantity'])
        df_transformed = cat_projector.fit_transform(df)

        # then
        df_expected = pd.DataFrame({'item': ['A', 'A', 'B', 'B', 'B'],
                                    'town': ['Paris', 'London', 'Paris', 'Paris', 'Roma'],
                                    'mean_price_per_item': [0.5, 0.5, 1.666667, 1.666667, 1.666667],
                                    'mean_quantity_per_item': [3.5, 3.5, 1.3333333333333333, 1.3333333333333333,
                                                               1.3333333333333333],
                                    'mean_price_per_town': [1.3333333333333333, 1.0, 1.3333333333333333,
                                                            1.3333333333333333, 1.0],
                                    'mean_quantity_per_town': [2, 4, 2, 2, 1]
                                    })

        df_transformed = df_transformed[df_expected.columns]

        assert_frame_equal(df_expected, df_transformed)

    def test_tagencoder_should_encode_categories_nicely(self):
        # given
        df = pd.DataFrame({'item': ['A', 'A', 'B', 'B', 'B']})

        # when
        tagencoder = TagEncoder()
        column_encoded = tagencoder.fit_transform(df['item'])

        # then
        column_expected = [0, 0, 1, 1, 1]

        self.assertListEqual(column_expected, column_encoded)

    def test_tagencoder_should_encode_unknown_categories_without_crashing_like_this_shitty_sklearn_labelencoder(self):
        # given
        df = pd.DataFrame({'item': ['A', 'A', 'B', 'B', 'B']})
        df_bis = pd.DataFrame({'item': ['A', 'A', 'B', 'C', 'B', 'D']})

        # when
        tagencoder = TagEncoder(missing_value=-1)
        tagencoder.fit(df['item'])
        column_encoded = tagencoder.transform(df_bis['item'])

        # then
        column_expected = [0, 0, 1, -1, 1, -1]

        self.assertListEqual(column_expected, column_encoded)


if __name__ == '__main__':
    unittest.main()
