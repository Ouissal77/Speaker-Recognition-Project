import  math
import numpy as np
import scipy.sparse as sp
from scipy.fftpack import dct

def rmvNan(A):
    cols_to_remove = np.any(np.isnan(A) | np.isinf(A), axis=0)
    new = A[:, ~cols_to_remove]
    return new