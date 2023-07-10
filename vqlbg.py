import numpy as np
from disteu import disteu
def vqlbg(d,k):
    e = .01
    r = np.mean(d, axis=1, keepdims=True)
    dpr = 10000

    for i in range(int(np.log2(k))):
        r = np.hstack((r * (1 + e), r * (1 - e)))
        while True:
            z = disteu(d, r)
            #m = np.min(z, axis=1)
            #ind = np.argmin(z, axis=1)
            ind = np.argmin(z, axis=1)
            m = z[np.arange(len(z)), ind]


            t = 0
            for j in range((2 ** (i+1)) ):
                r[:, j] = np.mean(d[:, ind == j], axis=1)
                smtg=r[:, j]
                smtg = smtg.reshape(len(smtg), 1)
                x = disteu(d[:, ind == j], smtg)

                #ind_j = np.where(m == j)[0]
              # r[:, j] = np.mean(d[:, ind_j], axis=1)
              # x = disteu(d[:, ind_j], r[:, j])
                for q in range(len(x)):
                    t += x[q]
            if (((dpr - t) / t) < e):
                break
            else:
                dpr = t


    return r
