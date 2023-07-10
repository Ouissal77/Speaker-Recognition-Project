import  math
import numpy as np
import scipy.sparse as sp
from scipy.fftpack import dct
from melfb import melfb
from disteu import disteu
def mfccMine(s,fs,nc):
    m = 100
    n = 256
    l = int(len(s)/nc)

    nbFrame = math.floor((l - n) / m) + 1
    M = np.zeros((n, nbFrame))

    for i in range(n) :
        for j in range(nbFrame) :
            M[i,j] = s[((j) * m) + i]

    h = np.hamming(n)
    M2 = np.dot(np.diag(h), M)
    print(M2.shape)

    frame = np.zeros((n, nbFrame), dtype=np.complex64)
    for i in range(nbFrame):
        frame[:, i] = np.fft.fft(M2[:, i])
    np.set_printoptions(precision=4, suppress=True)
    t = n / 2
    tmax = l / fs
    m = melfb(20, n, fs)
    m=m.toarray()
    m = m[1:, :]
    m = np.concatenate((m, m[:, 0].reshape(-1, 1)), axis=1)
    m = m[:, 1:]
  #  m = sp.csr_matrix(m)

    n2 = 1 + math.floor(n / 2)
    #n2 = 1 + int(np.floor(n / 2))
    abs_squared = np.abs(frame[:n2, :])
    abs_squared=np.square(abs_squared)
    # compute the product of m and the absolute value squared of frame
    z = m.dot(abs_squared)
    r=np.zeros_like(z)
    r=np.log(z)
    y = dct(r, axis=0, norm='ortho')

    return y



