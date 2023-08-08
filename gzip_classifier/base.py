from base64 import b64encode, b64decode
from gzip import compress, decompress
import json


class BaseClassifier(object):
    """ The base class of all classifiers. This object simply knows how to
    serialize and deserialize model data for use by other classification methods.
    """

    __slots__ = (
        '_model',
    )

    version = '1.0'

    def __init__(self):
        self._model = []

    def __repr__(self):
        size = len(self._model)
        name = type(self).__name__
        return f'{name}<size: {size}, ready: {self.is_ready}>'

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
    def model_settings(self):
        """ An optional value to serialize settings for this model
        into the model output for use later.

        Once imported from a source, these settings are re-applied to the
        classifier.
        """
        return {}

    def encode_row(self, row):
        def _encode(item):
            if isinstance(item, bytes):
                return b64encode(item).decode('utf-8')
            else:
                return b64encode(str(item).encode('utf-8')).decode('utf-8')
        return ' '.join([_encode(item) for item in row])

    @property
    def model(self):
        """ A writeable version of the data model for this classifier. """
        if not self.is_ready:
            raise ValueError('Cannot export un-trained model.')

        settings = [
            f'# Gzip Classifier Model Version {self.version}',
            '# This file contains model data with the following settings:',
            '#',
            f'# {json.dumps(self.model_settings)}',
        ]

        model_data = [
            self.encode_row(row)
            for row in self._model
        ]

        return '\n'.join([
            *settings,
            *model_data,
        ]).encode('utf-8')

    def decode_row(self, row: [bytes]):
        items = [b64decode(item) for item in row]
        return (
            items[0],
            items[1],
            float(items[2]),
            items[3].decode('utf-8'),
        )

    @model.setter
    def model(self, value: bytes):
        """ Update the data model for this classifier. """

        def _is_configuration(row: bytes):
            return row[0] == '#'

        def _configure(row: bytes):
            try:
                config = json.loads(row[1:])
            except json.JSONDecodeError:
                # This line does not contain model settings data. It could
                # be a comment.
                return

            for setting in self.model_settings:
                if value := config.get(setting):
                    setattr(self, setting, value)

        for row in value.decode('utf-8').split('\n'):
            if _is_configuration(row):
                self.decode_row(row)
            else:
                self._model.append(self.decode_row(row.split()))

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
