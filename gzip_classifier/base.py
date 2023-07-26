from base64 import b64encode, b64decode
from gzip import compress, decompress


class BaseClassifier(object):
    """ The base class of all classifiers. This object simply knows how to
    serialize and deserialize model data for use by other classification methods.
    """

    __slots__ = (
        '_model',
    )

    def __init__(self):
        self._model = []

    def __repr__(self):
        size = len(self._model)
        return f'Base<size: {size}, ready: {self.is_ready}>'

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
        raise NotImplementedError()

    def classify(self, sample, k=None):
        raise NotImplementedError()

    def classify_bulk(self, samples, k=None):
        raise NotImplementedError()

    @property
    def is_ready(self):
        return len(self._model) > 0

    @property
    def model(self):
        """ A writeable version of the data model for this classifier. """
        if not self.is_ready:
            raise ValueError('Cannot export un-trained model.')

        def _encode(item):
            return b64encode(item).decode('utf-8')

        return '\n'.join([
            ' '.join([_encode(item) for item in row])
            for row in self._model
        ]).encode('utf-8')

    @model.setter
    def model(self, value):
        """ Update the data model for this classifier. """
        self._model = [
            [b64decode(item).decode('utf-8') for item in row.split()]
            for row in value.split('\n')
        ]

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