import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cvx

def lowpass_2d(data, r=25):
    fs = np.fft.fft2(data)
    fltr = np.zeros_like(data, dtype=np.float)
    m, n = data.shape
    c = (m // 2, n // 2)
    if m % 2 == 0:
        di = 0
    else:
        di = 1
    if n % 2 == 0:
        dj = 0
    else:
        dj = 1
    y, x = np.ogrid[-c[0]:c[0] + di, -c[1]:c[1] + dj]
    mask = x ** 2 + y ** 2 <= r ** 2
    fltr[mask] = 1
    fs_filtered = np.fft.fftshift(np.multiply(np.fft.fftshift(fs), fltr))
    data_filtered = np.abs(np.fft.ifft2(fs_filtered))
    return data_filtered

def edge_find_1d(s1, tol=5e-2, ixs=None, ix0=0, w=30, mu=3, debug=False):
    # Returns the indices of edges in a 1-D array. This algorithm recursively segments the input array until all edges
    # have been found.
    if ixs is None:
        ixs = []
    x = cvx.Variable(len(s1))
    mu = cvx.Constant(mu)
    obj = cvx.Minimize(cvx.norm(s1[np.isfinite(s1)] - x[np.isfinite(s1)]) + mu * cvx.norm1(x[:-1] - x[1:]))
    prob = cvx.Problem(obj)
    prob.solve(solver='MOSEK')
    if debug:
        plt.plot(x.value)
        plt.show()
    s2 = np.abs(x.value[:-1] - x.value[1:])
    if debug:
        print(s2.max() - s2.min())
    if s2.max() - s2.min() < tol:
        # There are no step shifts in this data segment
        return ixs
    else:
        # There is a step shift in this data segment
        ix = np.argsort(-s2)[0]
        vr_best = -np.inf
        for j in range(ix - w, ix + w):
            jx = max(0, j)
            jx = min(jx, len(s1))
            sa = s1[:jx][np.isfinite(s1)[:jx]]
            sb = s1[jx:][np.isfinite(s1)[jx:]]
            vr = (np.std(s1[np.isfinite(s1)]) ** 2
                  - (len(sa) / len(s1[np.isfinite(s1)])) * np.std(sa)
                  - (len(sb) / len(s1[np.isfinite(s1)])) * np.std(sb))
            if vr > vr_best:
                vr_best = vr
                ix_best = jx
        ixs.append(ix_best + ix0)
        ixs1 = edge_find_1d(s1[:ix_best], tol=tol, ixs=ixs, ix0=ix0)
        ixs2 = edge_find_1d(s1[ix_best:], tol=tol, ixs=ixs1, ix0=ix0+ix_best)
        ixs2.sort()
        return ixs2
