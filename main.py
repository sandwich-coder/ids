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


dataset = 'cic17'
sampler = Sampler()

#load
loader = Loader()
array = loader.load(dataset)

#model
model = Autoencoder()

#train
trainer = Trainer()
trainer.train(array, model)

#test
out = model.flow(array)


# - plot -

os.makedirs('figures', exist_ok = True)
plot = Plot()
sampler = Sampler()
np.random.seed(seed = 1)    #standardized

#gradient descent
descent = plot.history(trainer)
descent.savefig('figures/history.png', dpi = 300)

#dashes
dashes = plot.dashes(array, model)
dashes.savefig('figures/dashes.png', dpi = 300)


# - anomaly detection (scan) -

normal = array.copy()
anomalous = sampler.sample(
    loader.load(dataset, normal = False),
    size = len(normal) // 100,
    )

#reconstruction errors
errors, error_metric = plot.errors(normal, anomalous, model, return_metric = True)
errors.savefig('figures/errors.png', dpi = 300)

contaminated = np.concatenate([
    normal,
    anomalous,
    ], axis = 0)

truth = np.zeros([len(contaminated)], dtype = 'int64')
truth[len(normal):] = 1
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

normal_test = loader.load(dataset, train = False)
anomalous_test = sampler.sample(
    loader.load(dataset, normal = False, train = False),
    size = len(normal_test) // 100,
    )

#reconstruction errors
errors_test = plot.errors(normal_test, anomalous_test, model)
errors_test.savefig('figures/errors-test.png', dpi = 300)

contaminated_test = np.concatenate([
    normal_test,
    anomalous_test,
    ], axis = 0)

truth_test = np.zeros([len(contaminated_test)], dtype = 'int64')
truth_test[len(normal_test):] = 1
truth_test = truth_test.astype('bool')

#Euclidean distance
error_test = error_metric(
    contaminated_test,
    model.flow(contaminated_test),
    )
prediction_test = np.where(error_test >= threshold, True, False)

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
