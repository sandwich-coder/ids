from basic import *
logger = logging.getLogger(name = __name__)

from torch.utils.data import DataLoader
from tqdm import tqdm


if torch.cuda.is_available():
    logger.info('CUDA is available.')
    device = torch.device('cuda')
    logger.info('GPU is assigned to \'device\'.')
else:
    device = torch.device('cpu')
    logger.info('CPU is assigned to \'device\' as fallback.')

learning_rate = 0.0001
epsilon = 1e-7
batch_size = 32
epochs = 100


class Trainer:
    """
    reference = [
        'device',
        'learning_rate',
        'epsilon',
        'batch_size',
        'epochs',
        ]
    """
    def __init__(self, Optimizer = optim.AdamW, LossFn = nn.MSELoss):
        if not issubclass(Optimizer, optim.Optimizer):
            raise TypeError('The optimizer should be a subclass of \'torch.nn.optim.Optimizer\'.')
        if not issubclass(LossFn, nn.Module):
            raise TypeError('\'LossFn\' should be a subclass of \'torch.nn.Module\'.')

        self.Optimizer = Optimizer
        self.loss_fn = LossFn()

        self.descent = None
        self.batchloss_final = None
        self.trained_array = None
        self.trained_model = None

    def __repr__(self):
        return 'trainer'
    
    def train(self, X, model):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The array must be tabular.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        X = X.copy()

        #processed
        data = model.process(X)

        #to gpu
        data = data.to(device)
        model.to(device)
        logger.info('\'device\' is allocated to \'data\' and \'model\'.')

        optimizer = self.Optimizer(
            model.parameters(),
            lr = learning_rate,
            eps = epsilon,
            )

        loader = DataLoader(
            data,
            batch_size = batch_size,
            shuffle = True,
            )
        self.descent = []
        logger.info('Training begins.')
        for lll in range(epochs):
            model.train()
            losses = []
            for t in tqdm(loader, leave = False, ncols = 70):

                output = model(t)
                loss = self.loss_fn(output, t)

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

                losses.append(loss.detach())    ###

            losses = torch.stack(losses, dim = 0)
            losses = losses.cpu()
            losses = losses.numpy()
            losses = losses.astype('float64')
            print('Epoch {epoch:>3} | loss: {loss_mean:<7}'.format(
                epoch = lll + 1,
                loss_mean = losses.mean(axis = 0, dtype = 'float64').round(decimals = 6),
                ))
            self.descent.append(losses)

        self.descent = np.concatenate(self.descent, axis = 0)
        self.batchloss_final = losses.mean(axis = 0, dtype = 'float64').tolist()
        self.trained_array = X.copy()
        self.trained_model = model
        logger.info(' - Training finished - ')

        #back to cpu
        model.cpu()
