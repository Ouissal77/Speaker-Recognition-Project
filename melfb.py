import numpy as np
import math
import scipy.sparse as sp
def melfb(p, n, fs):

    f0 = 700.0 / fs
    fn2 = math.floor(n / 2)

    lr = math.log(1 + 0.5 / f0) / (p + 1)

    # convert to fft bin numbers with 0 for DC term
    bl = n * (f0 * (np.exp(np.array([0, 1, p, p + 1]) * lr) - 1))

    b1 = math.floor(bl[0]) + 1
    b2 = math.ceil(bl[1])
    b3 = math.floor(bl[2])
    b4 = min(fn2, math.ceil(bl[3])) - 1
    pf = np.log(1 + np.arange(b1, b4+1) / n / f0) / lr
    fp = np.floor(pf).astype(int)
    pm = pf - fp



    r = np.concatenate((fp[b2-1:b4], 1 + fp[0:b3]))
    c = np.concatenate((np.arange(b2, b4+1), np.arange(1, b3+1)))+1
    v = 2 * np.concatenate((1 - pm[b2-1:b4], pm[0:b3]))

    m = sp.csr_matrix((v, (r, c)), shape=(p+1, 1 + fn2))
    #print(f'before :  {np.shape(m)}')
    #print(m)
   # m=m.toarray()
   # m=sp.coo_matrix(m[1:,1:])
#    print(f'after :  {np.shape(m)}')

   # m=sp.csr_matrix((v1, (r1, c1)), shape=(p, 1+fn2))
   # m.eliminate_zeros()

    return m

