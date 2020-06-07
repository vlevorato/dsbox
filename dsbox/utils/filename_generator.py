import random


class FilenameGenerator:
    """
    Class used to auto-generate temporary filenames used by intermediate operations. It adds an hexadecimal tag
    to a chosen prefix, like for instance 'X_1acf'. It keeps in memory all the generated hextags.

    Parameters
    ----------
    path: str
        temporary files path
    filename: str, default='X'
        prefix to temporary filenames
    hextag_length: int, default=4
        number of hexa digits used to generate file names
    sepchar: str, default='_'
        char separator between prefix filenames and hexa digits
    seed: int, default=123
        seed used by pseudo-random number generator

    Attributes
    ----------
    hextags: list
        contains all generated hexa tags

    """
    def __init__(self, path='', filename='X', hextag_length=4, sepchar='_', seed=123):
        self.hextags = []
        self.path = path
        self.filename = filename
        self.hextag_length = hextag_length
        self.sepchar = sepchar
        self.seed = seed
        random.seed(seed)

        self.max_val = int('F' * hextag_length, base=16)

    def generate_hextag(self):
        return hex(random.randint(0, self.max_val))[2:].rjust(self.hextag_length, '0')

    def generate_filename(self, pos=None):
        """
        Generates a random filename and keeps track of it by adding its hextag to hextags list

        Parameters
        ----------
        pos: int, default=None
            if None, the function generates a hextag, adds it to hextags list and returns generated filename.
            if not None, the function returns the previous generated filename with the pos index from the hextags list,
            but if pos gives an out of bounds index, it has the same behavior as when set to None.

        Returns
        -------
        :str
            generated filename path
        """


        if pos is None:
            hextag = self.generate_hextag()
            self.hextags.append(hextag)
        else:
            if pos >= len(self.hextags):
                for i in range(len(self.hextags), pos + 1):
                    self.hextags.append(self.generate_hextag())
            hextag = self.hextags[pos]

        return self.path + self.filename + self.sepchar + hextag
