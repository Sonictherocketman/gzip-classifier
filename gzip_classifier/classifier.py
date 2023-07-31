from collections import Counter
from gzip import compress
from multiprocessing import Pool

from .base import BaseClassifier
from .utils import (
    prepare_input,
    calc_distance,
    calc_distance_w_args,
    transform,
    transform_w_args,
)


class Classifier(BaseClassifier):
    """ A text-classification system that uses gzip to perform similarity
    comparisons and kNN to determine classification.

    This class implements the methods described in “Low-Resource” Text
    Classification: A Parameter-Free Classification Method with Compressors

    See https://arxiv.org/pdf/2212.09410.pdf for more information.
    """

    __slots__ = (
        '_model',
        'k',
    )

    def __init__(
        self,
        training_data=[],
        labels=[],
        k=1,
        auto_train=True,
        **kwargs
    ):
        self.k = k
        super().__init__(**kwargs)

        if auto_train and training_data and labels:
            self.train(training_data, labels)

    def __repr__(self):
        size = len(self._model)
        name = type(self).__name__
        return f'{name}<size: {size}, k: {self.k}, ready: {self.is_ready}>'

    @property
    def model_settings(self):
        return {**super().model_settings, 'k' : self.k}

    def train(self, training_data, labels):
        """ Given the set of training data and labels, train the model. """
        self._model = sorted((
            transform(item, label)
            for (item, label) in zip(training_data, labels)
        ), key=lambda x: x[2])

        if not self.is_ready:
            self._raise_invalid_configuration()

    def get_candidates(self, x1, Cx1, k):
        return sorted((
            (calc_distance(x1, Cx1, x2, Cx2), label)
            for x2, _, Cx2, label in self._model
        ), key=lambda x: x[0])

    def classify(self, sample, k=None, include_all=False):
        """ Attempt to classify the given text snippet using
        the training data and labels provided earlier.

        This method uses the method provided here:
        https://arxiv.org/pdf/2212.09410.pdf

        Broadly it takes a normalized distance measure between the training
        data and the provided sample, then finds the best match using a
        k-nearest-neighbor algorithm.

        k=1 is the ideal case (not suitable for real-world data). It is
        recommended to try out different integer k values to find one suitable
        for your data.

        Anecdotally, k=20 worked well for some uses.

        Set include_all=True to get the list of labels (length k) that were
        considered for classification.
        """
        k = k if k else self.k
        x1 = prepare_input(sample)
        Cx1 = len(compress(x1))
        candidates = self.get_candidates(x1, Cx1, k)
        return self._tabluate(candidates, k, include_all=include_all)

    def classify_bulk(self, samples, k=None, include_all=False):
        """ Perform classification on a list of text samples. """
        for sample in samples:
            yield self.classify(sample, k=k, include_all=include_all)

    @property
    def is_ready(self):
        return len(self._model) > 0

    def _raise_invalid_configuration(self):
        raise ValueError(
                'Cannot perform training. Missing or mismatched data. '
                'You must provide at least one item of training data and '
                'the number of training items must equal the number of labels.'
            )

    def _tabluate(self, results, k, include_all=False):
        top_k = results[:k]
        top_labels = list(label for (_, label) in top_k)
        most_common = Counter(top_labels).most_common(1)[0]
        if include_all:
            return (*most_common, top_labels)
        else:
            return most_common


class ParallelClassifier(Classifier):
    """ A version of the classic serial Classifier class that performs both
    training and classification in parallel using a process pool.

    For ease of use it is recommended to use this class within a context. This
    will ensure that all relevant internal state is cleaned up as needed.

    If not used within a context, then be sure to call the `start()` method before
    using the classifier and call the `close()` method when finished.

    Example:

        with ParallelClassifier() as classifier:
            classifier.train(data, labels)
            classifier.classify(sample, k)
    """

    __slots__ = (
        '_model',
        'k',
        'processes',
        'chunksize',
        'pool',
    )

    def __init__(
        self,
        processes=None,
        chunksize=1_000,
        **kwargs,
    ):
        self.processes = processes
        self.chunksize = chunksize
        self.pool = None
        super().__init__(**kwargs)

    def __enter__(self, processes=None):
        self.start(processes)
        return self

    def __exit__(self, type, value, traceback):
        self.close()

    def start(self, processes=None):
        """ Initialize the process worker pool. """
        self.processes = processes if processes else self.processes
        self.pool = Pool(processes=self.processes)

    def close(self):
        """ Shutdown the worker process pool. """
        try:
            self.pool.close()
        except Exception as e:
            self.pool.terminate()
            raise e

    def train(self, training_data, labels):
        """ Train the model as described in Classifier, but leveraging the
        process pool to improve performance.
        """
        if not self.pool:
            self._raise_invalid_pool()

        self._model = sorted(self.pool.imap(
            transform_w_args,
            zip(training_data, labels),
            self.chunksize,
        ), key=lambda x: x[2])

        if not self.is_ready:
            self._raise_invalid_configuration()

    def get_candidates(self, x1, Cx1, k):
        if not self.pool:
            self._raise_invalid_pool()

        values = (
            (x1, Cx1, x2, Cx2, label)
            for x2, _, Cx2, label in self._model
        )
        results = self.pool.imap(calc_distance_w_args, values, self.chunksize)
        return sorted(results, key=lambda x: x[0])

    def _raise_invalid_pool(self):
        raise ValueError(
                'The process pool has not yet been configured. Make sure to '
                'call the `start()` method prior to training or classifying.'
            )
