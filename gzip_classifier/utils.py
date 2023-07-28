from gzip import compress
from itertools import islice

from .types import Model, Index


def prepare_input(value: str):
    return value.lower().encode()


def calc_distance(x1: bytes, Cx1: int, x2: bytes, Cx2: int):
    x1_x2 = x1 + b' ' + x2
    Cx1_x2 = len(compress(x1_x2))
    return (Cx1_x2 - min(Cx1, Cx2)) / max(Cx1, Cx2)


def calc_distance_w_args(args):
    x1, Cx1, x2, Cx2, label = args
    return calc_distance(x1, Cx1, x2, Cx2), label


def transform(item: str, label: str):
    encoded_item = prepare_input(item)
    compressed_item = compress(encoded_item)
    return (
        encoded_item,
        compressed_item,
        len(compressed_item),
        label,
    )


def transform_w_args(args: (str, str)):
    return transform(*args)


def batched(iterable: iter, n: int):
    """Batch data into tuples of length n. The last batch may be shorter."""
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while batch := tuple(islice(it, n)):
        yield batch


def generate_index(model: Model, bin_size: int = 1_000) -> Index:
    """ Generate an index that allows for quick and easy searching of the
    model based on lengths. The index essentially 'bins' the data into chunks
    which can be used to more precisely search the model rather than searching
    everything.

    The implementation of this chunking is likely very relevant to the performance
    of the quick search algorithm.

    TODO: Improve Performance

    Eventually this should instead batch the data by chunks of std
    deviation since these naive chunks are likely very similar in composition
    especially if the lengths of each item in the training set is very similar.
    """
    index = []
    i_start = 0
    for chunk in batched(model, bin_size):
        first, last = chunk[0], chunk[-1]
        index.append((
            (i_start, i_start + bin_size),
            (first[2], last[2])
        ))
        i_start += bin_size
    return index


def get_likely_bin(index: Index, Cx1: int):
    """ Identify and return the indicies of the first bin that is likely
    to contain the items of the training set that are most similar to the
    item provided.
    """
    first_positions, first_lengths = index[0]
    if Cx1 < first_lengths[0]:
        # In this case the input length is shorter than any item
        # in the training set, so use the first bin to match against.
        return first_positions

    last_positions, last_lengths = index[0]
    if Cx1 > last_lengths[0]:
        # In this case the input length is longer than any item
        # in the training set, so use the last bin to match against.
        return last_positions

    # Ideally the sample length with be within the boundaries of the
    # training set and so we need to search for the first bin that
    # contains the items of that length.
    return [
        positions
        for positions, lengths in index
        if Cx1 >= lengths[0] and Cx1 < lengths[1]
    ][0]
