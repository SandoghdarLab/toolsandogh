import itertools
import math
from multiprocessing import Pool

import numpy as np

from toolsandogh.iscat_analysis import SharedArray


def test_SharedArray():
    dtypes = [np.float32, np.complex64, np.int32]
    shapes: list[tuple[int, ...]] = [(2,), (2, 3), (2, 3, 5)]
    for dtype in dtypes:
        for shape in shapes:
            # Test individual array access
            sa = SharedArray(shape, dtype=dtype)
            indices = itertools.product(*[list(range(dim)) for dim in shape])
            for index in indices:
                value = dtype(np.random.randint(256))
                sa[*index] = value
                assert sa[*index] == value

            # Fill array with zeros
            sa[...] = dtype(0)

            # Test concurrent access
            size = math.prod(shape)
            with Pool(processes=size) as pool:

                def initialize(array, index, value):
                    array[*index] = value

                pool.starmap(
                    initialize, [(sa, i, dtype(n)) for n, i in enumerate(indices)]
                )
            for n, index in enumerate(indices):
                value = dtype(n)
                assert sa[*index] == value
