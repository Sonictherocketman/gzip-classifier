from gzip import compress
from itertools import islice
from statistics import stdev

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


def generate_simple_index(model: Model, bin_size: int = 1_000) -> Index:
    """ Generate an index that allows for quick and easy searching of the
    model based on lengths. The index essentially 'bins' the data into chunks
    which can be used to more precisely search the model rather than searching
    everything.

    The implementation of this chunking is likely very relevant to the performance
    of the quick search algorithm.
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


def add_percent_overscan(start: int, end: int, bound: int, overscan: float = None):
    """ Return new start and end values that are extended by the overscan
    percentage. These numbers are bounded by 0 and the bound such that all
    values are "0 <= x <= bound".

    Example:

    # To add an overscan of 20% to each end of the boundary consider the
    # following example:
    > add_overscan(100, 200, bound=225, overscan=0.2)
    > (55, 225)
    """
    if overscan is None:
        return start, end

    padding = int(bound * overscan)
    return max(0, start - padding), min(bound, end + padding)


def generate_stdev_index(model: Model) -> (Index, float):
    """ Return an index for the given model as well as the standard deviation
    for the given model's compressed length.

    This index generation process takes no configuration as it uses the standard
    deviation to create the indicies. The chunks are any contiguous section of the
    model of at least one standard deviation in length (though they may be longer
    if two chunks would end with the same length value).
    """
    index = []
    i_start = 0
    deviation = round(stdev((row[2] for row in model)))

    last_group = None
    for chunk in batched(model, deviation):
        first, last = chunk[0], chunk[-1]
        current_group = (
            (i_start, i_start + deviation),
            (first[2], last[2])
        )

        if not last_group:
            # Just started.
            index.append(current_group)
            last_group = current_group
        elif current_group[1][1] == last_group[1][1]:
            # The ending value is the same. Extend the last group.
            last_group = (
                (last_group[0][0], current_group[0][1]),
                (last_group[1][0], current_group[1][1]),
            )
        else:
            # This is a new group. Move forward.
            if last_group not in index:
                index.append(last_group)
            last_group = current_group

        i_start += deviation
    return index, deviation


def add_stdev_overscan(start: int, end: int, bound: int, overscan=True, stdev: int = None):
    """ Return new start and end values that are extended by the standard
    deviation. These numbers are bounded by 0 and the bound such that all
    values are "0 <= x <= bound".

    Supplying a false value for `overscan` will skip this process and return
    the provided start, end values.

    Example:

    # To add an overscan of 20% to each end of the boundary consider the
    # following example:
    > add_overscan(100, 200, bound=225, stdev=10)
    > (90, 210)

    """
    if not stdev:
        return start, end

    return max(0, start - stdev), min(bound, end + stdev)


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

    last_positions, last_lengths = index[-1]
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
        if Cx1 >= lengths[0] and Cx1 <= lengths[1]
    ][0]
