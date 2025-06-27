from basic import *
logger = logging.getLogger(name = __name__)

import pandas as pd


class Loader:
    def __init__(self):
        pass
    def __repr__(self):
        return 'loader'

    def load(self, name, normal = True, train = True):
        if not isinstance(name, str):
            raise TypeError('The name should be a string.')
        if not isinstance(normal, bool):
            raise TypeError('\'normal\' should be boolean.')
        if not isinstance(train, bool):
            raise TypeError('\'train\' should be boolean.')
        if train:
            kind = 'train'
        else:
            kind = 'test'

        orderless = [
            'attack_flag',
            'attack_name',
            'attack_step',
            'destination port',
            ]

        df = pd.read_csv('datasets/{name}/{kind}-flow.csv'.format(
                name = name,
                kind = kind,
                ))
        if normal:
            df = df[df['attack_flag'] == 0]
        else:
            df = df[df['attack_flag'] == 1]
        df = df.drop(columns = orderless)
        
        array = df.to_numpy(dtype = 'float64', copy = True)
        array = (array - array.min()) / (array.max() - array.min())
        array = (array - np.float64(0.5)) * np.float64(2)

        return array
