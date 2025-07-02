from basic import *
logger = logging.getLogger(name = __name__)

from sklearn.preprocessing import MinMaxScaler


class Autoencoder(nn.Module):
    def __init__(self, input_dim = 77):
        if not isinstance(input_dim, int):
            raise TypeError('The input dim should be an integer.')
        if not input_dim > 0:
            raise ValueError('The input dim must be positive.')
        super().__init__()
        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Sequential(nn.Linear(self.input_dim, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, 33), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(33, 9), nn.Dropout(0.1), nn.Tanh()),
            )

        self.decoder = nn.Sequential(
            nn.Sequential(nn.Linear(9, 33), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(33, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, 100), nn.Dropout(0.1), nn.GELU()),
            nn.Sequential(nn.Linear(100, self.input_dim), nn.Dropout(0.1), nn.Tanh()),
            )

        #initialized
        with torch.no_grad():
            nn.init.xavier_uniform_(self.encoder[-1][0].weight)
            nn.init.xavier_uniform_(self.decoder[-1][0].weight)

    def __repr__(self):
        return 'autoencoder'

    def forward(self, t):
        if not t.size(dim = 1) == self.input_dim:
            raise ValueError('The number of features must be {input_dim}.'.format(
                input_dim = self.input_dim,
                ))    # Checking of the number of features should be placed in the 'forward' instead of the 'process' and 'unprocess'.
        t = torch.clone(t)

        t = self.encoder(t)
        t = self.decoder(t)

        return t


    def process(self, X, train = True):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not X.ndim == 2:
            raise ValueError('The input must be tabular.')
        if not X.dtype == np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
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


    # This method solely aims to be the inverse. It doesn't add any other functionality.
    def unprocess(self, processed):
        if not isinstance(processed, torch.Tensor):
            raise TypeError('The input should be a \'torch.Tensor\'.')
        if processed.requires_grad:
            raise ValueError('The input must not be on the graph. \nThis method doesn\'nt automatically detach such Tensors.')
        if not processed.dim() == 2:
            raise ValueError('The input must be tabular.')
        if not processed.dtype == torch.float32:
            logger.warning('The dtype doesn\'t match.')
            processed = processed.to(torch.float32)
        processed = torch.clone(processed)

        _ = processed.numpy()
        unprocessed = _.astype('float64')
        unprocessed = self.fit_scaler.inverse_transform(unprocessed)
        return unprocessed


    def flow(self, X):
        if not isinstance(X, np.ndarray):
            raise TypeError('The input should be a \'numpy.ndarray\'.')
        if not X.ndim == 2:
            raise ValueError('The input must be tabular.')
        if not X.dtype == np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        X = X.copy()

        self.eval()

        Y = self.process(X, train = False)
        Y = self.forward(Y)
        Y = Y.detach()    ###
        Y = self.unprocess(Y)

        return Y
