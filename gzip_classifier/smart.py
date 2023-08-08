from gzip import compress

from .naive import ParallelNaiveClassifier
from .quick import QuickClassifier
from .utils import (
    prepare_input,
    calc_distance,
    calc_distance_w_args,
    generate_quantile_index,
    add_percent_overscan,
    get_likely_bin,
)


class SmartClassifier(QuickClassifier):
    """ A classifier similar to the QuickClassifier that groups data by
    quantiles.
    """

    def __init__(self, *args, quantiles=10, **kwargs):
        self.quantiles = quantiles
        super().__init__(*args, **kwargs)

    def __repr__(self):
        size = len(self._model)
        name = type(self).__name__
        return (
            f'{name}<size: {size}, k: {self.k}, quantiles: {self.quantiles}, '
            f'ready: {self.is_ready}>'
        )

    @property
    def model_settings(self):
        return {**super().model_settings, 'quantiles' : self.quantiles}

    def get_indicies(self, Cx1, overscan):
        return add_percent_overscan(
            *get_likely_bin(self._index, Cx1),
            bound=len(self._model),
            overscan=overscan,
        )

    def train(self, training_data, labels):
        super().train(training_data, labels)
        self._index = generate_quantile_index(self._model, self.quantiles)


class SmartParallelClassifier(SmartClassifier, ParallelNaiveClassifier):
    """ This classifier implements the same smart search algorithm based on the
    training distribution as the SmartClassifier, but also provides the ability
    to spread the computation along multiple cores for both training and testing.
    """

    def get_candidates(self, x1, Cx1, k, overscan=None):
        start, stop = self.get_indicies(Cx1, overscan)
        values = (
            (x1, Cx1, x2, Cx2, label)
            for x2, _, Cx2, label in self._model[start:stop]
        )
        results = self.pool.imap(calc_distance_w_args, values, self.chunksize)
        return sorted(results, key=lambda x: x[0])
