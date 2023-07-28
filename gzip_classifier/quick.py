from gzip import compress

from .classifier import Classifier
from .utils import (
    prepare_input,
    calc_distance,
    generate_index,
    get_likely_bin,
    add_overscan,
)


class QuickClassifier(Classifier):

    def __init__(self, *args, bin_size=1_000, overscan=None, **kwargs):
        self.bin_size = bin_size
        self.overscan = overscan
        super().__init__(*args, **kwargs)

    # TODO: Add import/export or recalc for importing models.

    def train(self, training_data, labels):
        super().train(training_data, labels)
        self._index = generate_index(self._model)

    def classify(self, sample, k=None, include_all=True, overscan=None):
        k = k if k else self.k

        x1 = prepare_input(sample)
        Cx1 = len(compress(x1))

        start, stop = add_overscan(
            *get_likely_bin(self._index, Cx1),
            bound=len(self._model),
            overscan=self.overscan,
        )
        candidates = sorted((
            (calc_distance(x1, Cx1, x2, Cx2), label)
            for x2, _, Cx2, label in self._model[start:stop]
        ), key=lambda x: x[0])

        return self._tabluate(candidates, k, include_all=include_all)
