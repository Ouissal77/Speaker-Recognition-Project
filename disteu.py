import numpy as np

def disteu(x,y):
    M, N = x.shape
    M2, P = y.shape

    if M != M2:
        raise ValueError('Matrix dimensions do not match.')

    d = np.zeros((N, P))

    if N < P:
        copies = np.zeros(P, dtype=int)
        for n in range(N):
            d[n, :] = np.sum((x[:, n + copies] - y) ** 2, axis=0)
    else:
        copies = np.zeros(N, dtype=int)
        for p in range(P):
            d[:, p] = np.sum((x - y[:, p + copies]) ** 2, axis=0)

    d = np.sqrt(d)

    return d