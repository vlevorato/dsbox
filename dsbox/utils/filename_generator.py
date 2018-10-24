import os
import binascii


class FilenameGenerator:
    def __init__(self, path='', filename='X', hextag_length=4, sepchar='_'):
        self.hextags = set()
        self.path = path
        self.filename = filename
        self.hextag_length = hextag_length
        self.sepchar = sepchar

    def generate_hextag(self):
        return binascii.b2a_hex(os.urandom(self.hextag_length)).decode("utf-8")

    def generate_filename(self):
        hextag = self.generate_hextag()
        while hextag in self.hextags:
            hextag = self.generate_hextag()

        self.hextags.add(hextag)
        return self.path + self.filename + self.sepchar + hextag
