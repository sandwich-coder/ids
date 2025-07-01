from basic import *
logger = logging.getLogger(name = __name__)

import pandas as pd
import seaborn as sb

from tools.sampler import Sampler

def _to_frame(array):
    feature = np.arange(1, array.shape[1]+1, dtype = 'int64')
    feature = feature.repeat(array.shape[0], axis = 0)
    value = array.transpose().reshape([-1]).copy()
    frame = np.stack([feature, value], axis = 1)
    frame = pd.DataFrame(frame, columns = ['feature', 'value'])
    frame = frame.astype({'feature':'int64'})
    return frame

sampler = Sampler()


class Plotter:
    """
    reference = [
        _to_frame,
        sampler,
        ]
    """
    def __init__(self):
        pass
    def __repr__(self):
        return 'plot'

    def before_after(self, X, ae, index = None):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The array must be tabular')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if index is not None:
            if not isinstance(index, np.ndarray):
                raise TypeError('The indices should be a \'numpy.ndarray\'.')
            if index.dtype != np.int64:
                raise ValueError('The indices must be of \'numpy.int64\'.')
            if index.ndim != 1:
                raise ValueError('The indices must be 1-dimensional.')
            index = index.copy()
        else:
            index = np.random.choice(len(X), size = 30, replace = False)
        X = X.copy()

        before = X.copy()
        after = ae.flow(X)

        figs = []
        for lll in index:
            fig = pp.figure(layout = 'constrained', figsize = (10, 5))
            gs = fig.add_gridspec(nrows = 1, ncols = 2)

            ax_1 = fig.add_subplot(gs[1-1])
            ax_1.set_box_aspect(1)
            ax_1.set_aspect(1)
            ax_1.set_xticks([])
            ax_1.set_yticks([])
            plot_1 = ax_1.imshow(
                before[lll].reshape([28, 28]),
                cmap = 'grey',
                vmin = -1, vmax = 1,
                )

            ax_2 = fig.add_subplot(gs[2-1])
            ax_2.set_box_aspect(1)
            ax_2.set_aspect(1)
            ax_2.set_xticks([])
            ax_2.set_yticks([])
            plot_2 = ax_2.imshow(
                after[lll].reshape([28, 28]),
                cmap = 'grey',
                vmin = -1, vmax = 1,
                )

            figs.append(fig)

        return figs


    def errors(self, normal, anomalous, ae):
        if not isinstance(normal, np.ndarray):
            raise TypeError('The normal should be a \'numpy.ndarray\'.')
        if not isinstance(anomalous, np.ndarray):
            raise TypeError('The anomalous should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if not (normal.ndim == anomalous.ndim == 2):
            raise ValueError('The arrays must be tabular.')
        if not (normal.shape[1] == anomalous.shape[1]):
            raise ValueError('The normal and anomalous must have the same number of features.')
        if not normal.dtype == np.float64:
            logger.warning('The dtype doesn\'t match.')
            normal = normal.astype('float64')
        if not anomalous.dtype == np.float64:
            logger.warning('The dtype doesn\'t match.')
            anomalous = anomalous.astype('float64')
        normal = normal.copy()
        anomalous = anomalous.copy()
        fig = pp.figure(layout = 'constrained')
        ax = fig.add_subplot()
        ax.set_box_aspect(1)
        ax.set_title('Reconstruction Errors')
        ax.set_xticks([])
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        #Euclidean distance
        def diff(X, Y):
            error = (Y - X) ** 2
            error = error.sum(axis = 1, dtype = 'float64')
            error = np.sqrt(error, dtype = 'float64')
            return error

        normal_error = diff(
            normal,
            ae.flow(normal),
            )
        anomalous_error = diff(
            anomalous,
            ae.flow(anomalous),
            )

        temp = 25 / len(normal_error) ** 0.5
        if temp > 1:
            temp = 1
        plot_1 = ax.plot(
            np.linspace(0, 1, num = len(normal_error), dtype = 'float64'), normal_error,
            marker = 'o', markersize = 3 * temp,
            linestyle = '',
            alpha = 0.8,
            color = 'tab:blue',
            label = 'normal',
            )
        temp = 25 / len(anomalous_error) ** 0.5
        if temp > 1:
            temp = 1
        plot_2 = ax.plot(
            np.linspace(0, 1, num = len(anomalous_error), dtype = 'float64'), anomalous_error,
            marker = 'o', markersize = 3 * temp,
            linestyle = '',
            alpha = 0.8,
            color = 'tab:red',
            label = 'anomalous',
            )

        ax.legend()
        return fig


    def dashes(self, X, ae, sample = True, size = 300):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if not isinstance(sample, bool):
            raise TypeError('\'sample\' should be boolean.')
        if not isinstance(size, int):
            raise TypeError('\'size\' should be an integer.')
        if not X.ndim == 2:
            raise ValueError('The array must be tabular.')
        if not size > 0:
            raise ValueError('\'size\' must be positive.')
        if not X.dtype == np.float64:
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        X = X.copy()

        if not sample:
            sample = X.copy()
        else:
            sample = sampler.sample(X, size)

        fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.5)
        ax.set_title('Dashes   (#samples: {count})'.format(
            count = len(sample),
            ))
        ax.set_xlabel('feature #')
        ax.set_ylabel('value')
        pp.setp(ax.get_yticklabels(), ha = 'right', va = 'center', rotation = 90)

        compressed = ae.process(sample, train = False)
        compressed = ae.encoder(compressed)
        compressed = compressed.detach()    ###
        compressed = compressed.numpy()
        compressed = compressed.astype('float64')

        plots = []
        index = range(len(compressed))
        for ll in index:
            instance = compressed[ll]

            plot = ax.plot(
                range(1, 1+len(instance)), instance,
                marker = 'o', markersize = 3 / len(compressed) ** 0.5,
                linestyle = '--', linewidth = 3 / len(compressed),
                alpha = 0.8,
                color = 'tab:orange',
                )
            plots.append(plot)

        ax.set_xticks(np.arange(1, 1+compressed.shape[1], dtype = 'int64'))

        return fig


    def boxes(self, X, ae, sample = True, size = 300):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if not isinstance(sample, bool):
            raise TypeError('\'sample\' should be boolean.')
        if not isinstance(size, int):
            raise TypeError('\'size\' should be an integer.')
        if not X.ndim == 2:
            raise ValueError('The array should be tabular.')
        if not size > 0:
            raise ValueError('The sample size must be positive.')
        if not X.dtype == 'float64':
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        X = X.copy()        

        if not sample:
            sample = X.copy()
        else:
            sample = sampler.sample(X, size)

        fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.5)
        ax.set_title('Boxes   (#samples: {count})'.format(
            count = len(sample),
            ))
        ax.set_xlabel('feature #')
        ax.set_ylabel('value')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        compressed = ae.process(sample, train = False)
        compressed = ae.encoder(compressed)
        compressed = compressed.detach()    ###
        compressed = compressed.numpy()
        compressed = compressed.astype('float64')

        sb.boxplot(
            data = _to_frame(compressed),
            x = 'feature', y = 'value',
            orient = 'x',
            whis = (0, 100),
            color = 'tab:orange',
            ax = ax,
            )

        return fig


    def violins(self, X, ae, sample = True, size = 300):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if not isinstance(ae, nn.Module):
            raise TypeError('The autoencoder should be a \'torch.nn.Module\'.')
        if not isinstance(sample, bool):
            raise TypeError('\'sample\' should be boolean.')
        if not isinstance(size, int):
            raise TypeError('\'size\' should be an integer.')
        if not X.ndim == 2:
            raise ValueError('The array should be tabular.')
        if not size > 0:
            raise ValueError('The sample size must be positive.')
        if not X.dtype == 'float64':
            logger.warning('The dtype doesn\'t match.')
            X = X.astype('float64')
        X = X.copy()

        if not sample:
            sample = X.copy()
        else:
            sample = sampler.sample(X, size)

        fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.5)
        ax.set_title('Violins   (#samples: {count})'.format(
            count = len(sample),
            ))
        ax.set_xlabel('feature #')
        ax.set_ylabel('value')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        compressed = ae.process(sample, train = False)
        compressed = ae.encoder(compressed)
        compressed = compressed.detach()    ###
        compressed = compressed.numpy()
        compressed = compressed.astype('float64')

        sb.violinplot(
            data = _to_frame(compressed),
            x = 'feature', y = 'value',
            orient = 'x',
            bw_adjust = 0.5,
            inner = 'quart',
            hue = None, color = 'deepskyblue',
            density_norm = 'width',
            ax = ax,
            )

        return fig
