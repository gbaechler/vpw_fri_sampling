from __future__ import division
import numpy as np
from scipy import linalg
import os
from matplotlib import rcParams

# for latex rendering
os.environ['PATH'] = os.environ['PATH'] + ':/usr/texbin' + ':/opt/local/bin' + ':/Library/TeX/texbin/'
rcParams['text.usetex'] = True
rcParams['text.latex.unicode'] = True


def distance(x1, x2):
    """
    Given two arrays of numbers x1 and x2, pairs the cells that are the
    closest and provides the pairing matrix index: x1(index(1,:)) should be as
    close as possible to x2(index(2,:)). The function outputs the average of the
    absolute value of the differences abs(x1(index(1,:))-x2(index(2,:))).
    :param x1: vector 1
    :param x2: vector 2
    :return: d: minimum distance between d
             index: the permutation matrix
    """
    x1 = np.reshape(x1, (1, -1), order='F')
    x2 = np.reshape(x2, (1, -1), order='F')
    N1 = x1.size
    N2 = x2.size
    diffmat = np.abs(x1 - np.reshape(x2, (-1, 1), order='F'))
    min_N1_N2 = np.min([N1, N2])
    index = np.zeros((min_N1_N2, 2), dtype=int)
    if min_N1_N2 > 1:
        for k in xrange(min_N1_N2):
            d2 = np.min(diffmat, axis=0)
            index2 = np.argmin(diffmat, axis=0)
            index1 = np.argmin(d2)
            index2 = index2[index1]
            index[k, :] = [index1, index2]
            diffmat[index2, :] = float('inf')
            diffmat[:, index1] = float('inf')
        d = np.mean(np.abs(x1[:, index[:, 0]] - x2[:, index[:, 1]]))
    else:
        d = np.min(diffmat)
        index = np.argmin(diffmat)
        if N1 == 1:
            index = np.array([1, index])
        else:
            index = np.array([index, 1])
    return d, index


def periodic_sinc(t, M):
    numerator = np.sin(t)
    denominator = M * np.sin(t / M)
    idx = np.abs(denominator) < 1e-12
    numerator[idx] = np.cos(t[idx])
    denominator[idx] = np.cos(t[idx] / M)
    return numerator / denominator


def Tmtx(data, K):
    """Construct convolution matrix for a filter specified by 'data'
    """
    return linalg.toeplitz(data[K::], data[K::-1])


def Tmtx_ri_half(data, K, D):
    """
    We assume the data that is used to build the T matrix is arranged in the folloing way:
    the first half is real part of the POSITIVE freqeuncy; the second half is the imaginary
    part of the POSITIVE frequency. Imaginary part at frequency zero is NOT stored in the
    data vector.
    :param data: data used to build the T matrix
    :param K: number of Dirac
    :return:
    """
    data_len = data.size
    assert data_len % 2 == 1
    real_pos_len = np.int((data_len + 1) / 2)
    data_ri = np.dot(D, data)
    data_r = data_ri[:real_pos_len]
    data_i = data_ri[real_pos_len:]
    Tr = linalg.toeplitz(data_r[K::], data_r[K::-1])
    Ti = linalg.toeplitz(data_i[K::], data_i[K::-1])
    return np.vstack((np.hstack((Tr, -Ti)), np.hstack((Ti, Tr))))


# def expansion_mtx(len_data):
#     """
#     :param len_data: length of the first half REAL data
#     :return:
#     """
#     D0 = np.eye(len_data)
#     D1 = np.vstack((D0[:-1, ::-1], D0))
#     D2 = np.vstack((-D0[:-1, ::-1], D0))
#     D2 = D2[:, 1:]
#     return D1, D2


def expansion_mtx_pos(len_data_pos):
    """
    :param len_data: length of the first half REAL data
    :return:
    """
    D1 = np.eye(len_data_pos)
    D2 = D1[:, 1:]
    return linalg.block_diag(D1, D2)


def Rmtx(coef, K, seq_len):
    """A dual convolution matrix of Tmtx. Use the commutativness of a convolution:
    a * b = b * c
    Here seq_len is the INPUT sequence length
    """
    col = np.concatenate(([coef[-1]], np.zeros(seq_len - K - 1)))
    row = np.concatenate((coef[::-1], np.zeros(seq_len - K - 1)))
    return linalg.toeplitz(col, row)


def Rmtx_ri_half(coef, K, seq_len_pos, D):
    """A dual convolution matrix of Tmtx. Use the commutativness of a convolution:
    a * b = b * c
    Here seq_len is the INPUT sequence length
    """
    assert coef.dtype == 'float'
    assert coef.size == 2 * (K + 1)
    coef_r = coef[:K+1]
    coef_i = coef[K+1:]
    col_r = np.concatenate(([coef_r[-1]], np.zeros(seq_len_pos - K - 1)))
    row_r = np.concatenate((coef_r[::-1], np.zeros(seq_len_pos - K - 1)))
    col_i = np.concatenate(([coef_i[-1]], np.zeros(seq_len_pos - K - 1)))
    row_i = np.concatenate((coef_i[::-1], np.zeros(seq_len_pos - K - 1)))
    R_r = linalg.toeplitz(col_r, row_r)
    R_i = linalg.toeplitz(col_i, row_i)
    return np.dot(np.vstack((np.hstack((R_r, -R_i)), np.hstack((R_i, R_r)))), D)


def iqml_recon(G, a, K, noise_level, max_ini=100, stop_cri='mse'):
    compute_mse = (stop_cri == 'mse')
    M = G.shape[1]
    GtG = np.dot(G.conj().T, G)
    Gt_a = np.dot(G.conj().T, a)

    max_iter = 50
    min_error = float('inf')
    # beta = linalg.solve(GtG, Gt_a)
    beta = linalg.lstsq(G, a)[0]

    Tbeta = Tmtx(beta, K)
    rhs = np.concatenate((np.zeros(2 * M + 1), [1.]))
    rhs_bl = np.concatenate((Gt_a, np.zeros(M - K)))

    for ini in xrange(max_ini):
        c = np.random.randn(K + 1) + 1j * np.random.randn(K + 1)
        c0 = c.copy()
        error_seq = np.zeros(max_iter)
        R_loop = Rmtx(c, K, M)

        for loop in xrange(max_iter):
            Mtx_loop = np.vstack((np.hstack((np.zeros((K + 1, K + 1)), Tbeta.conj().T, np.zeros((K + 1, M)),
                                             c0[:, np.newaxis])),
                                  np.hstack((Tbeta, np.zeros((M - K, M - K)), -R_loop, np.zeros((M - K, 1)))),
                                  np.hstack((np.zeros((M, K + 1)), -R_loop.conj().T, GtG, np.zeros((M, 1)))),
                                  np.hstack((c0[np.newaxis].conj(), np.zeros((1, 2 * M - K + 1))))
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop = (Mtx_loop + Mtx_loop.conj().T) / 2.
            clvlambda = linalg.solve(Mtx_loop, rhs)
            c = clvlambda[0:K + 1]

            R_loop = Rmtx(c, K, M)

            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.conj().T)),
                                    np.hstack((R_loop, np.zeros((M - K, M - K))))
                                    ))
            # matrix should be Hermitian symmetric
            Mtx_brecon = (Mtx_brecon + Mtx_brecon.conj().T) / 2.
            bl = linalg.solve(Mtx_brecon, rhs_bl)
            b_recon = bl[0:M]

            error_seq[loop] = linalg.norm(a - np.dot(G, b_recon))
            if error_seq[loop] < min_error:
                min_error = error_seq[loop]
                b_opt = b_recon
                c_opt = c
            if min_error < noise_level and compute_mse:
                break
        if min_error < noise_level and compute_mse:
            break

    return b_opt, min_error, c_opt, ini


def iqml_recon_ri(G, a, K, noise_level, max_ini=100, stop_cri='mse'):
    """
    Here we assume both the measurements a and the linear transformation G are real-valued
    """
    # make sure the assumption (real-valued) is true
    assert not np.iscomplexobj(a) and not np.iscomplexobj(G)
    compute_mse = (stop_cri == 'mse')
    GtG = np.dot(G.T, G)
    Gt_a = np.dot(G.T, a)

    two_Lp1 = G.shape[1]
    assert two_Lp1 % 2 == 1
    L = np.int((two_Lp1 - 1) / 2.)

    sz_T0 = 2 * (L - K + 1)
    sz_T1 = 2 * (K + 1)
    sz_G1 = 2 * L + 1
    sz_R0 = sz_T0
    sz_R1 = 2 * L + 1
    sz_coef = 2 * (K + 1)

    D = expansion_mtx_pos(L + 1)

    max_iter = 50
    min_error = float('inf')
    beta = linalg.lstsq(G, a)[0]

    Tbeta = Tmtx_ri_half(beta, K, D)  # has 2(K+1) columns
    rhs = np.concatenate((np.zeros(sz_T1 + sz_T0 + sz_G1), [1.]))
    rhs_bl = np.concatenate((Gt_a, np.zeros(sz_R0)))

    for ini in xrange(max_ini):
        c_ri = np.random.randn(2 * (K + 1))  # first half of c_ri: c_real, second half: c_imag
        c0 = c_ri.copy()
        error_seq = np.zeros(max_iter)
        # R has (2L + 1) columns
        R_loop = Rmtx_ri_half(c_ri, K, L + 1, D)

        for loop in xrange(max_iter):
            Mtx_loop = np.vstack((np.hstack((np.zeros((sz_T1, sz_T1)), Tbeta.T, np.zeros((sz_T1, sz_R1)),
                                             c0[:, np.newaxis])),
                                  np.hstack((Tbeta, np.zeros((sz_T0, sz_T0)), -R_loop, np.zeros((sz_T0, 1)))),
                                  np.hstack((np.zeros((sz_R1, sz_T1)), -R_loop.T, GtG, np.zeros((sz_G1, 1)))),
                                  np.hstack((c0[np.newaxis, :], np.zeros((1, sz_T0 + sz_R1 + 1))))
                                  ))
            # matrix should be Hermitian symmetric
            Mtx_loop = (Mtx_loop + Mtx_loop.T) / 2.
            c_ri = linalg.solve(Mtx_loop, rhs)[:sz_coef]

            R_loop = Rmtx_ri_half(c_ri, K, L + 1, D)

            Mtx_brecon = np.vstack((np.hstack((GtG, R_loop.T)),
                                    np.hstack((R_loop, np.zeros((sz_R0, sz_R0))))
                                    ))
            # matrix should be Hermitian symmetric
            Mtx_brecon = (Mtx_brecon + Mtx_brecon.T) / 2.
            b_recon_ri = linalg.solve(Mtx_brecon, rhs_bl)[:sz_G1]

            error_seq[loop] = linalg.norm(a - np.dot(G, b_recon_ri))
            if error_seq[loop] < min_error:
                min_error = error_seq[loop]
                b_opt = b_recon_ri[:L+1] + 1j * np.concatenate((np.array([0]),
                                                                b_recon_ri[L+1:]))
                c_opt = c_ri[:K+1] + 1j * c_ri[K+1:]
            if min_error < noise_level and compute_mse:
                break
        if min_error < noise_level and compute_mse:
            break

    return b_opt, min_error, c_opt, ini

if __name__ == '__main__':
    pass

    # K = 3
    # L = 8
    # real_len = L + 1
    # full_len = 2 * L + 1
    # b = np.random.randn(full_len)
    # coef = np.random.randn((K + 1) * 2)
    # D = expansion_mtx_pos(real_len)
    # Tb = Tmtx_ri_half(b, K, D)
    # Rc = Rmtx_ri_half(coef, K, real_len, D)
    # res1 = np.dot(Tb, coef)
    # res2 = np.dot(Rc, b)
    # print linalg.norm(res1 - res2)
