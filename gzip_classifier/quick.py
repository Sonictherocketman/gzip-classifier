from .naive import NaiveClassifier, ParallelNaiveClassifier
from .utils import (
    prepare_input,
    calc_distance,
    calc_distance_w_args,
    generate_simple_index,
    add_percent_overscan,
    get_likely_bin,
    compress,
)


class QuickClassifier(NaiveClassifier):
    """ A serial classifier that uses indexing & binning to avoid searching
    the entire training set.

    This classifier is significantly faster at classifying test samples than
    the normal Classifier (and even the Parallel Classifier) at the cost of
    some extra effort during training and a potential loss of accuracy.

    The binning method used may result in lower accuracy than a full data scan
    and may not be suitable for all use cases.

    This classifier allows for a variable performance improvement of roughly
    N = length of training data / bin size, such that N = 10 results in a 10x
    performance improvement at the cost of increasingly inaccurate results at
    higher N.

    The `overscan` parameter should be a number between 0 and 1. The larger the
    value of the overscan, the less performance gain will be had.

    Implementation Notes
    --------------------

    During training the sorted model data is indexed into groups of length
    `bin_size`. At classification-time, the sample is compared with the index
    to find a bin that most likely contains the training data most similar to
    the input sample.

    This process means some samples are close to the edge of a bin boundary
    and so simply searching the bin exclusively can lead to poor classification
    accuracy. To remedy this issue, use the `overscan` parameter.

    Overscanning adjusts the boundaries of the bin scan to a certain percentage
    beyond the boundaries of the given bin.

    For example:

    A model containing 1,000 training data items and a bin size of 100 would be
    chunked into 10 bins each containing 100 items. An overscan value of 5% (0.05)
    would result in a scan of 200 total items (100 items from the most relevant
    bin plus 50 items on either side of that bin's boundaries).

    This overscan 'fuzzes' the boundaries of each bin and allows for higher
    accuracy when compared to a naive binned scanning method.
    """

    def __init__(self, *args, bin_size=1_000, overscan=None, **kwargs):
        self.bin_size = bin_size
        self.overscan = overscan
        super().__init__(*args, **kwargs)

    def __repr__(self):
        size = len(self._model)
        name = type(self).__name__
        return (
            f'{name}<size: {size}, k: {self.k}, bin_size: {self.bin_size}, '
            f'ready: {self.is_ready}>'
        )

    @property
    def model_settings(self):
        return {
            **super().model_settings,
            'overscan': self.overscan,
            'bin_size': self.bin_size,
        }

    def get_indicies(self, Cx1, overscan):
        return add_percent_overscan(
            *get_likely_bin(self._index, Cx1),
            bound=len(self._model),
            overscan=overscan,
        )

    def train(self, training_data, labels):
        super().train(training_data, labels)
        self._index = generate_simple_index(self._model, self.bin_size)

    def get_candidates(self, x1, Cx1, k, overscan=None):
        start, stop = self.get_indicies(Cx1, overscan)
        return sorted((
            (calc_distance(x1, Cx1, x2, Cx2), label)
            for x2, _, Cx2, label in self._model[start:stop]
        ), key=lambda x: x[0])

    def classify(self, sample, k=None, include_all=True, overscan=None):
        k = k if k else self.k
        overscan = overscan if overscan else self.overscan
        x1 = prepare_input(sample)
        Cx1 = len(compress(x1))
        candidates = self.get_candidates(x1, Cx1, k, overscan)
        return self._tabluate(candidates, k, include_all=include_all)


class QuickParallelClassifier(QuickClassifier, ParallelNaiveClassifier):
    """ This classifier leverages the bin scanning and overscan method
    implemented in the QuickClassifier but also leverages multi-core processing
    to perform the comparisons within a bin resulting in a roughly N times
    performance improvement (N = # of cores).
    """

    def get_candidates(self, x1, Cx1, k, overscan=None):
        start, stop = self.get_indicies(Cx1, overscan)
        values = (
            (x1, Cx1, x2, Cx2, label)
            for x2, _, Cx2, label in self._model[start:stop]
        )
        results = self.pool.imap(calc_distance_w_args, values, self.chunksize)
        return sorted(results, key=lambda x: x[0])
