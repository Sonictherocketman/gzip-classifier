import csv
import os

import numpy as np
import pytest

from gzip_classifier import Classifier


THIS_DIR = os.path.dirname(os.path.realpath(__file__))
FIXTURES_DIR = os.path.join(THIS_DIR, 'fixtures')


fields = (
    'source',
    'url',
    'title',
    'image',
    'category',
    'description',
    'rank',
    'pubdate',
    'video',
)


def get_set(filename, text_column, label_column):
    with open(filename) as f:
        reader = csv.DictReader(
            f,
            fields,
            delimiter='\t',
            quoting=csv.QUOTE_NONE,
        )
        return [
            (row[text_column], row[label_column])
            for row in reader
            if row[text_column] and row[label_column]
        ]


@pytest.fixture
def test_data():
    test_set = np.array(get_set(
        os.path.join(FIXTURES_DIR, 'test.txt'),
        'description',
        'category'
    ))
    return [description for description, _ in test_set]

@pytest.fixture
def training_data():
    training_set = np.array(get_set(
        os.path.join(FIXTURES_DIR, 'train.txt'),
        'description',
        'category'
    ))
    return [description for description, _ in training_set]

@pytest.fixture
def labels():
    training_set = np.array(get_set(
        os.path.join(FIXTURES_DIR, 'train.txt'),
        'description',
        'category'
    ))
    return [label for _, label in training_set]


def test_classify(test_data, training_data, labels):
    classifier = Classifier()
    classifier.train(training_data, labels)
    result = classifier.classify_bulk(test_data, k=10)
    assert len([r for r in result]) > 0


def test_serlialize(test_data, training_data, labels):
    classifier = Classifier()
    classifier.train(training_data, labels)

    model = classifier.model
    assert len(model) > 0


def test_deserlialize(test_data, training_data, labels):
    classifier = Classifier()
    classifier.train(training_data, labels)

    model = classifier.model

    classifier2 = Classifier.using_model(model)
    assert classifier2.is_ready


def test_file_inout_deserlialize(test_data, training_data, labels):
    classifier = Classifier(training_data, labels, auto_train=True)

    with open('model.txt', 'wb') as f:
        f.write(classifier.model)

    with open('model.txt', 'rb') as f:
        classifier2 = Classifier.using_model(f.read())
        classifier2.is_ready

    assert str(classifier) == str(classifier2)


def test_file_inout_deserlialize_compact(test_data, training_data, labels):
    classifier = Classifier(training_data, labels, auto_train=True)

    with open('model.txt.gz', 'wb') as f:
        f.write(classifier.compact_model)

    with open('model.txt.gz', 'rb') as f:
        classifier2 = Classifier.using_compact_model(f.read())
        classifier2.is_ready

    assert str(classifier) == str(classifier2)
