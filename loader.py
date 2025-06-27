from basic import *
logger = logging.getLogger(name = __name__)

import pandas as pd


class Loader:
    def __init__(self):
        pass
    def __repr__(self):
        return 'loader'

    def load(self, name, train = True, normal = True):
        if not isinstance(name, str):
            raise TypeError('The name should be a string.')
        if not isinstance(train, bool):
            raise TypeError('\'train\' should be boolean.')
        if not isinstance(normal, bool):
            raise TypeError('\'normal\' should be boolean.')
        if train:
            kind = 'train'
        else:
            kind = 'test'
        if normal:
            label = 'benign'
        else:
            label = 'attack'

        df = pd.read_csv('datasets/{name}/{kind}-flow.csv'.format(
                name = name,
                kind = kind,
                ))
