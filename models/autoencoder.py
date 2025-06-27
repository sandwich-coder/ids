from basic import *
logger = logging.getLogger(name = __name__)

from sklearn.preprocessing import MinMaxScaler


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(77, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, 33), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(33, 11), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(11, 3), nn.Dropout(0.1), nn.Tanh()),
            )

        self.decoder = nn.Sequential(
            nn.Sequential(nn.Linear(3, 11), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(11, 33), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(33, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, 77), nn.Dropout(0.1), nn.Tanh()),
            )

        #initialized
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder[-1][0].weight)
            nn.init.xavier_uniform_(self.decoder[-1][0].weight)

    def __repr__(self):
        return 'autoencoder'

    def forward(self, t):
        if t.size(dim = 1) != 77:
            raise ValueError('The number of features must be 77.')    # Checking of the number of features should be placed in the 'forward' instead of the 'process' and 'unprocess'.
        t = torch.clone(t)

        """
        t = t.reshape([-1, 1, 28, 28])
        """

        t = self.encoder(t)
        t = self.decoder(t)

        return t


    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        X = X.copy()

        if not train:
            pass
        else:
            scaler = MinMaxScaler(feature_range = (-1, 1))
            scaler.fit(X)
            self.fit_scaler = scaler

        processed = self.fit_scaler.transform(X)
        processed = torch.tensor(processed, dtype = torch.float32)
        return processed


    # This method solely aims to be the inverse of the 'process'. It doesn't add any other functionality.
    def unprocess(self, T):
        if not isinstance(T, torch.Tensor):
            raise TypeError('The input should be a \'torch.Tensor\'.')
        if T.requires_grad:
            raise ValueError('The input must not be on the graph. \nThis method doesn\'nt automatically detach such Tensors.')
        if T.dtype != torch.float32:
            T = T.to(torch.float32)
        if T.dim() != 2:
            raise ValueError('The input must be tabular.')
        T = torch.clone(T)

        _ = T.numpy()
        unprocessed = _.astype('float64')
        unprocessed = self.fit_scaler.inverse_transform(unprocessed)
        return unprocessed


    def flow(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The input must be tabular.')
        X = X.copy()

        self.eval()

        X = self.process(X, train = False)
        X = self.forward(X)
        X = X.detach()    ###
        X = self.unprocess(X)

        return X
