#! /usr/bin/env python3
#
# Method from https://arxiv.org/pdf/2212.09410.pdf
# Data from http://groups.di.unipi.it/~gulli/AG_corpus_of_news_articles.html
#
# Get newsSpace from URL above and generate files as shown below:
#   head -100  newsSpace | tail -25 > test.txt
#   tail -100000 newsSpace > train.txt

from collections import Counter
import csv
import gzip
import numpy as np


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


test_set = np.array(get_set('test.txt', 'description', 'category'))
training_set = np.array(get_set('train.txt', 'description', 'category'))
compressed_training_set = [
    len(gzip.compress(x2.encode()))
    for x2, _ in training_set
]
k = 20  # no idea why. but 20 seems to work best.


for (x1, _) in test_set:
    Cx1 = len(gzip.compress(x1.encode()))
    distance_from_x1 = []
    for (x2_, Cx2) in zip(training_set, compressed_training_set):
        x2, _ = x2_
        x1x2 = " ".join([x1, x2])
        Cx1x2 = len(gzip.compress(x1x2.encode()))
        ncd = (Cx1x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)
        distance_from_x1.append(ncd)
    sorted_idx = np.argsort(np.array(distance_from_x1))
    top_k_class = training_set[sorted_idx[:k], 1]
    predict_class = Counter(top_k_class).most_common(1)
    print(f'{x1=}, {predict_class=}')
