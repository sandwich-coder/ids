from basic import *
logger = logging.getLogger(name = __name__)

from sklearn.preprocessing import MinMaxScaler


class Autoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_dim = None
        self.latent_dim = None
        self.encoder = None
        self.decoder = None
        self.scaler = None
    def __repr__(self):
        return 'autoencoder'

    def build(self, input_dim, latent_dim):
        if not isinstance(input_dim, int):
            raise TypeError('The input dim should be an integer.')
        if not isinstance(latent_dim, int):
            raise TypeError('The latent dimension should be an integer.')
        if not input_dim > 0:
            raise ValueError('The input dim must be positive.')
        if not latent_dim > 0:
            raise ValueError('The latent dimension must be positive.')
        if not latent_dim < 30:
            raise ValueError('The layers are configured only for the latent dimension lower than 50.')
        if not latent_dim <= input_dim:
            raise ValueError('The latent dimension must be smaller than or same as the input.')

        encoder = nn.Sequential(
            nn.Sequential(nn.Linear(input_dim, 100), nn.GELU()),
            nn.Sequential(nn.Linear(100, 30), nn.GELU()),
            nn.Sequential(nn.Linear(30, latent_dim), nn.Tanh()),
            )

        decoder = nn.Sequential(
            nn.Sequential(nn.Linear(latent_dim, 30), nn.GELU()),
            nn.Sequential(nn.Linear(30, 100), nn.GELU()),
            nn.Sequential(nn.Linear(100, input_dim), nn.Tanh()),
            )

        #initialized
        with torch.no_grad():
            nn.init.xavier_uniform_(encoder[-1][0].weight)
            nn.init.xavier_uniform_(decoder[-1][0].weight)

        #stored
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = encoder
        self.decoder = decoder


    def forward(self, t):
        if None in (self.input_dim, self.latent_dim, self.encoder, self.decoder):
            raise NotImplementedError('The model has not been built.')
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
            if self.scaler is None:
                raise NotImplementedError('The scaler has not been made.')
        else:
            scaler = MinMaxScaler(feature_range = (-1, 1))
            scaler.fit(X)
            self.scaler = scaler    #stored

        processed = self.scaler.transform(X)
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
        unprocessed = self.scaler.inverse_transform(unprocessed)
        return unprocessed


    #process->inference->unprocess
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
