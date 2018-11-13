import unittest

from dsbox.utils import FilenameGenerator


class TestFilenameGenerator(unittest.TestCase):
    def test_generate_filename(self):

        # given
        fg = FilenameGenerator()

        # when
        filename_10 = fg.generate_filename(pos=10)
        filename_9 = fg.generate_filename(pos=9)

        # then
        expected_filename_10 = 'X_1a92'
        expected_filename_9 = 'X_ae7d'

        self.assertEquals(expected_filename_10, filename_10)
        self.assertEquals(expected_filename_9, filename_9)

if __name__ == '__main__':
    unittest.main()
