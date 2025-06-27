import sys, os, subprocess

#python check
if sys.version_info[:2] != (3, 10):
    raise RuntimeError('This module is intended to be run on Python 3.10.')
else:
    print('Python version checked')

from basic import *
logging.basicConfig(level = 'INFO')
logger = logging.getLogger(name = __name__)

from sklearn.metrics import precision_score, recall_score, f1_score

from loader import Loader
from models.autoencoder import Autoencoder
from trainer import Trainer
from sampler import Sampler
from plot import Plot

#gpu driver check
sh = 'nvidia-smi'
sh_ = subprocess.run('which ' + sh, shell = True, capture_output = True, text = True)
if sh_.stdout == '':
    logger.info('Command \'{command}\' does not exist.'.format(command = sh))
else:
    sh_ = subprocess.run(
        sh,
        shell = True, capture_output = True, text = True,
        )
    cuda_version = sh_.stdout.split()
    cuda_version = cuda_version[cuda_version.index('CUDA') + 2]
    if torch.version.cuda is None:
        logger.info('The installed pytorch is not built with CUDA. Install a CUDA-enabled.')
    elif float(cuda_version) < float(torch.version.cuda):
        logger.info('The supported CUDA is lower than installed. Upgrade the driver.')
    else:
        logger.info('Nvidia driver checked')


#load
loader = Loader()
array_train = loader.load('cic23')
array_test = loader.load('cic23', train = False)

#model
model = Autoencoder()

#train
trainer = Trainer()
trainer.train(array_train, model)

#test
out_train = model.flow(array_train)
out_test = model.flow(array_test)


# - plot -

os.makedirs('figures', exist_ok = True)
plot = Plot()
sampler = Sampler()
np.random.seed(seed = 1)    #standardized

normal = array_train.copy()
normal = sampler.sample(normal, size = 30000)
anomalous = loader.load('cloths')
anomalous = sampler.sample(anomalous, size = 30000)

#gradient descent
descent = plot.history(trainer)
descent.savefig('figures/history.png', dpi = 300)

#dashes
dashes = plot.dashes(normal, model)
dashes.savefig('figures/dashes.png', dpi = 300)

#reconstruction errors
errors, error_metric = plot.errors(normal, anomalous, model, return_metric = True)
errors.savefig('figures/errors.png', dpi = 300)


# - anomaly detection (scan) -

contaminated = np.concatenate([
    sampler.sample(normal, size = 27000),
    sampler.sample(anomalous, size = 3000),
    ], axis = 0)

truth = np.zeros([30000], dtype = 'int64')
truth[27000:] = 1
truth = truth.astype('bool')

# The threshold is determined manually by observing the error plot.
errors.show()
threshold = input('threshold: ')
threshold = float(threshold)

error = error_metric(
    contaminated,
    model.flow(contaminated),
    )
prediction = np.where(error >= threshold, True, False)

print('\n\n')
print('     precision (train): {precision}'.format(
    precision = precision_score(truth, prediction),
    ))
print('        recall (train): {recall}'.format(
    recall = recall_score(truth, prediction),
    ))
print('            F1 (train): {f1}'.format(
    f1 = f1_score(truth, prediction),
    ))


# - anomaly detection (test) -

contaminated = np.concatenate([
    sampler.sample(
        loader.load('digits', train = False),
        size = 27000,
        ),
    sampler.sample(
        loader.load('cloths', train = False),
        size = 3000,
        ),
    ], axis = 0)

truth = np.zeros([30000], dtype = 'int64')
truth[27000:] = 1
truth = truth.astype('bool')

#Euclidean distance
error = error_metric(
    contaminated,
    model.flow(contaminated),
    )
prediction = np.where(error >= threshold, True, False)

print('\n\n')
print('      precision (test): {precision}'.format(
    precision = precision_score(truth, prediction),
    ))
print('         recall (test): {recall}'.format(
    recall = recall_score(truth, prediction),
    ))
print('             F1 (test): {f1}'.format(
    f1 = f1_score(truth, prediction),
    ))
