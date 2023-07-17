from base64 import b64encode, b64decode
from collections import Counter
from gzip import compress, decompress
import numpy as np


class Classifier(object):
    """ A text-classification system that uses gzip to perform similarity
    comparisons and kNN to determine classification.

    This class implements the methods described in “Low-Resource” Text
    Classification: A Parameter-Free Classification Method with Compressors

    See https://arxiv.org/pdf/2212.09410.pdf for more information.
    """

    __slots__ = (
        'training_data',
        'training_data_gz',
        'training_data_lengths',
        'labels',
        'k',
    )

    def __init__(
        self,
        training_data=[],
        labels=[],
        k=1,
        auto_train=False
    ):
        self.training_data = []
        self.training_data_gz = []

        self.k = k
        self.labels = np.array([])
        self.training_data_lengths = np.array([])

        if auto_train:
            self.train(training_data, labels)

    def __repr__(self):
        size = len(self.training_data)
        return f'Classifier<size: {size}, ready: {self.is_ready}>'

    @classmethod
    def using_model(cls, model):
        """ Create an instance of this classifier using the provided serialized
        model. This model must have been created using the `model` attribute.
        """
        instance = cls()
        instance.model = model
        return instance

    @classmethod
    def using_compact_model(cls, compact_model):
        """ Create an instance of this classifier using the provided serialized
        compact model. This model must have been created using the
        `compact_model` attribute.
        """
        instance = cls()
        instance.compact_model = compact_model
        return instance

    def train(self, training_data, labels):
        """ Given the set of training data and labels, train the model. """
        self.training_data = np.array(training_data)
        self.labels = np.array(labels)

        if not self.is_ready:
            self._raise_invalid_configuration()

        self.training_data_gz = [
            compress(item) for item in self.training_data
        ]

        self.training_data_lengths = [
            len(item) for item in self.training_data_gz
        ]

    def classify(self, sample, k=None):
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
        """
        k = k if k else self.k
        Cx1 = len(compress(sample.encode()))
        distance = []
        for (x2, Cx2) in zip(self.training_data, self.training_data_lengths):
            x1_x2 = " ".join([sample, x2])
            Cx1_x2 = len(compress(x1_x2.encode()))
            ncd = (Cx1_x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
            distance.append(ncd)
        sorted_idx = np.argsort(np.array(distance))
        top_k_class = self.labels[sorted_idx[:self.k]]
        return Counter(top_k_class).most_common(1)[0]

    def classify_bulk(self, samples, k=None):
        """ Perform classification on a list of text samples. """
        for sample in samples:
            yield self.classify(sample, k=k)

    @property
    def is_ready(self):
        return (
            len(self.training_data) == len(self.labels)
            and len(self.labels) > 0
        )

    @property
    def is_trained(self):
        return self.training_data_lengths

    @property
    def model(self):
        """ A writeable version of the data model for this classifier. """
        if not self.is_trained:
            raise ValueError('Cannot export un-trained model.')

        def _encode(item):
            return b64encode(item).decode('utf-8')

        return '\n'.join([
            f'{_encode(item)} {_encode(label)}'
            for item, label in zip(self.training_data_gz, self.labels)
        ]).encode('utf-8')

    @model.setter
    def model(self, value):
        """ Update the data model for this classifier. """
        def _decode(item):
            return b64decode(item)

        training_data_lengths, labels = [], []

        for row in value.decode('utf-8').split('\n'):
            data, label = row.split(' ')
            item = _decode(data)
            item_gz = decompress(item)

            self.training_data.append(item)
            self.training_data_gz.append(item_gz)
            training_data_lengths.append(len(item_gz))
            labels.append(_decode(label))

        self.training_data_lengths = np.array(training_data_lengths)
        self.labels = np.array(labels)

    @property
    def compact_model(self):
        """ The same as `model` but the returned value is already gzipped for
        easy storage on disk.
        """
        return compress(self.model)

    @compact_model.setter
    def compact_model(self, value):
        """ Set the model to a value exported from `compact_model`. """
        self.model = decompress(value)

    def _raise_invalid_configuration(self):
        raise ValueError(
                'Cannot perform training. Missing or mismatched data. '
                'You must provide at least one item of training data and '
                'the number of training items must equal the number of labels.'
            )
