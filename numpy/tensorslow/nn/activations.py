import tensorslow as slow
import numpy as np


def relu(x):
    # TODO: Don't like this
    return slow.maximum(slow.tensor(np.zeros(x.shape)), x)
