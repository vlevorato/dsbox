import random


class FilenameGenerator:
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

        if pos is None:
            hextag = self.generate_hextag()
            self.hextags.append(hextag)
        else:
            if pos >= len(self.hextags):
                for i in range(len(self.hextags), pos + 1):
                    self.hextags.append(self.generate_hextag())
            hextag = self.hextags[pos]

        return self.path + self.filename + self.sepchar + hextag
