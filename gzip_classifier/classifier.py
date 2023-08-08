from base64 import b64decode
from collections import Counter
from gzip import compress
from itertools import groupby
from multiprocessing import Pool

from .naive import NaiveClassifier, ParallelNaiveClassifier
from .utils import (
    batched,
    prepare_input,
    calc_distance_v2,
    transform_v2,
)


class Classifier(NaiveClassifier):

    def __init__(
        self,
        chunksize=20,
        dictionary_size=int(1e10),
        **kwargs
    ):
        self.chunksize = chunksize
        self.dictionary_size = dictionary_size
        super().__init__(**kwargs)

    def __repr__(self):
        size = len(self._model)
        name = type(self).__name__
        ready = 'ready' if self.is_ready else 'not ready'
        chunksize = self.chunksize
        return f'{name}<n: {size}, k: {self.k}, chunksize: {chunksize}, {ready}>'

    @property
    def model_settings(self):
        return {
            **super().model_settings,
            'chunksize': self.chunksize,
            'dictionary_size': self.dictionary_size,
            'tally_method': self.tally_method,
        }

    def encode_row(self, row):
        compressor, rest = row[0], row[1:]
        # TODO: We need to persist the compressor header.
        # Which is not available in the default compressor object.
        return super().encode_row([header, *rest])

    def decode_row(self, row:[bytes]):
        items = [b64decode(item) for item in row]
        return (
            items[0],
            items[1].decode('utf-8'),
        )

    def train(self, training_data, labels):
        self._model = [
            transform_v2(item, label, self.dictionary_size)
            for (item, label) in self._group_and_sort(training_data, labels)
        ]

        if not self.is_ready:
            self._raise_invalid_configuration()

    def get_candidates(self, sample, k):
        return (
            (calc_distance_v2(sample, compressor.copy()), label)
            for compressor, label in self._model
        )

    def classify(self, sample, k=None, include_all=False):
        k = k if k else self.k
        x1 = prepare_input(sample)
        candidates = self.get_candidates(x1, k)
        return self._tabluate(candidates, k, include_all=include_all)

    def _group_and_sort(self, training_data, labels):
        sorted_data = sorted(zip(training_data, labels), key=lambda x: x[1])
        grouped_data = groupby(sorted_data, lambda x: x[1])
        chunked_groups = (
            (batched((text for text, _ in data), self.chunksize), label)
            for (label, data) in grouped_data
        )

        return (
            ('\n'.join(set(chunk)), label)
            for (chunks, label) in chunked_groups
            for chunk in chunks
        )


class ParallelClassifier(ParallelNaiveClassifier):
    pass
#     """ A version of the classic serial Classifier class that performs both
#     training and classification in parallel using a process pool.
#
#     For ease of use it is recommended to use this class within a context. This
#     will ensure that all relevant internal state is cleaned up as needed.
#
#     If not used within a context, then be sure to call the `start()` method before
#     using the classifier and call the `close()` method when finished.
#
#     Example:
#
#         with ParallelClassifier() as classifier:
#             classifier.train(data, labels)
#             classifier.classify(sample, k)
#     """
#
#     __slots__ = (
#         '_model',
#         'k',
#         'processes',
#         'chunksize',
#         'pool',
#     )
#
#     def __init__(
#         self,
#         processes=None,
#         **kwargs,
#     ):
#         self.processes = processes
#         self.pool = None
#         super().__init__(**kwargs)
#
#         """ Train the model as described in NaiveClassifier, but leveraging the
#         process pool to improve performance.
#         """
#         if not self.pool:
#             self._raise_invalid_pool()
#
#         self._model = sorted(self.pool.imap(
#             transform_w_args,
#             self._group_and_sort(training_data, labels),
#             self.chunksize,
#         ), key=lambda x: x[2])
#
#         if not self.is_ready:
#             self._raise_invalid_configuration()
#
#     def get_candidates(self, x1, Cx1, k):
#         if not self.pool:
#             self._raise_invalid_pool()
#
#         values = (
#             (x1, Cx1, x2, Cx2, label)
#             for x2, _, Cx2, label in self._model
#         )
#         results = self.pool.imap(calc_distance_w_args, values, self.chunksize)
#         return sorted(results, key=lambda x: x[0])
