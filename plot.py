from basic import *
logger = logging.getLogger(name = __name__)

import pandas as pd
import seaborn as sb


class Plot:
    def __init__(self):
        pass
    def __repr__(self):
        return 'plot'

    def history(self, trainer):
        fig = pp.figure(layout = 'constrained', figsize = (10, 7.3))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.7)
        ax.set_title('Descent')
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot = ax.plot(
            np.arange(1, trainer.descent.shape[0]+1, dtype = 'int64'), trainer.descent,
            marker = 'o', markersize = 0.3,
            linestyle = '--', linewidth = 0.1,
            color = 'slategrey',
            label = 'final: {final}'.format(
                final = round(trainer.batchloss_final, ndigits = 4),
                )
            )
        ax.legend()

        return fig


    def dashes(self, X, model, sample = True, size = 300):
        if not isinstance(X, np.ndarray):
            raise TypeError('The array should be a \'numpy.ndarray\'.')
        if X.dtype != np.float64:
            X = X.astype('float64')
        if X.ndim != 2:
            raise ValueError('The array must be tabular.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        if not isinstance(sample, bool):
            raise TypeError('\'sample\' should be boolean.')
        if not isinstance(size, int):
            raise TypeError('\'size\' should be an integer.')
        if size < 1:
            raise ValueError('\'size\' must be positive.')
        X = X.copy()

        if not sample:
            sample = X.copy()
        else:
            sample = np.random.choice(np.arange(
                len(X),
                ), size = size, replace = False)
            sample = X[sample]

        compressed = model.process(sample, train = False)
        compressed = model.encoder(compressed)
        compressed = compressed.detach()    ###
        compressed = compressed.numpy()
        compressed = compressed.astype('float64')

        fig = pp.figure(layout = 'constrained', figsize = (10, 5.4))
        ax = fig.add_subplot()
        ax.set_box_aspect(0.5)
        ax.set_title('Dashes   (#samples: {count})'.format(
            count = len(compressed),
            ))
        ax.set_xlabel('feature #')
        ax.set_ylabel('value')
        pp.setp(ax.get_yticklabels(), ha = 'right', va = 'center', rotation = 90)
        plots = []
        index = range(len(compressed))
        for ll in index:
            instance = compressed[ll]

            plot = ax.plot(
                range(1, 1+len(instance)), instance,
                marker = 'o', markersize = 45 / (compressed.shape[0] ** 0.5 * compressed.shape[1]),
                linestyle = '--', linewidth = 30 / (compressed.shape[0] * compressed.shape[1]),
                color = 'tab:orange',
                alpha = 0.5,
                )
            plots.append(plot)

        ax.set_xticks(np.arange(1, 1+compressed.shape[1], dtype = 'int64'))

        return fig


    def errors(self, normal, anomalous, model, return_metric = False):
        if not isinstance(normal, np.ndarray):
            raise TypeError('The normal should be a \'numpy.ndarray\'.')
        if normal.dtype != np.float64:
            normal = normal.astype('float64')
        if normal.ndim != 2:
            raise ValueError('The normal must be tabular.')
        if not isinstance(anomalous, np.ndarray):
            raise TypeError('The anomalous should be a \'numpy.ndarray\'.')
        if anomalous.dtype != np.float64:
            anomalous = anomalous.astype('float64')
        if anomalous.ndim != 2:
            raise ValueError('The anomalous must be tabular.')
        if not isinstance(model, nn.Module):
            raise TypeError('The model should be a \'torch.nn.Module\'.')
        if not isinstance(return_metric, bool):
            raise TypeError('\'return_metric\' should be boolean.')
        normal = normal.copy()
        anomalous = anomalous.copy()

        def diff(in_, out_):
            error = (out_ - in_) ** 2
            error = error.sum(axis = 1, dtype = 'float64')
            error = np.sqrt(error, dtype = 'float64')
            return error

        normal_error = diff(
            normal,
            model.flow(normal),
            )
        anomalous_error = diff(
            anomalous,
            model.flow(anomalous),
            )

        fig = pp.figure(layout = 'constrained')
        ax = fig.add_subplot()
        ax.set_box_aspect(1)
        ax.set_title('Reconstruction Errors')
        ax.set_xticks([])
        pp.setp(ax.get_yticklabels(), rotation = 90, ha = 'right', va = 'center')

        plot_1 = ax.plot(
            np.linspace(0, 1, num = len(normal_error), dtype = 'float64'), normal_error,
            marker = 'o', markersize = 3 / len(normal_error) ** 0.5,
            alpha = 0.8,
            linestyle = '',
            color = 'tab:blue',
            label = 'normal',
            )
        plot_2 = ax.plot(
            np.linspace(0, 1, num = len(anomalous_error), dtype = 'float64'), anomalous_error,
            marker = 'o', markersize = 3 / len(anomalous_error) ** 0.5,
            alpha = 0.8,
            linestyle = '',
            color = 'red',
            label = 'anomalous',
            )
        ax.axhline(
            np.median(normal_error),
            marker = '',
            linestyle = 'dashed',
            color = 'black',
            label = 'median (normal)',
            )
        ax.axhline(
            np.median(anomalous_error),
            marker = '',
            linestyle = 'dotted',
            color = 'black',
            label = 'median (anomalous)',
            )

        ax.legend()

        if return_metric:
            return fig, diff
        else:
            return fig
