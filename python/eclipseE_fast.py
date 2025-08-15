import timeit
import numpy as np

def eclipseE_fast(weights, alpha, beta):
    '''
        This function ...
            Args: ...
            Outputs: ...
    '''
    # length
    l = len(weights)
    
    alpha, beta = 0.0, 1.0
    p = alpha * beta
    m = (alpha + beta) / 2
    trivial_Lip_sq = 1

    time_begin = timeit.default_timer()
    for i in range(1, l):
        di = weights['w'+str(i)].shape[0]
        Wi = weights['w'+str(i)]

        Xi_prev = Xi if i > 1 else np.identity(weights['w'+str(i)].shape[1])
        Inv_Xi_prev = np.linalg.inv(Xi_prev)

        mat = Wi @ Inv_Xi_prev @ Wi.T
        eigvals, eigvecs = np.linalg.eig(mat)
        li = 1 / (2 * m**2 * np.max(eigvals))
        Xi = li * np.identity(di) - li**2 * m**2 * mat

        # calculate the trivial lip
        trivial_Lip_sq *= np.linalg.norm(Wi)**2

    Wl = weights['w'+str(l)]
    eigvals, eigvecs = np.linalg.eig(Wl.T @ Wl @ np.linalg.inv(Xi))
    oneoverF = np.max(eigvals)
    Lip_sq_est = oneoverF
    Lip_est = np.sqrt(Lip_sq_est)
    
    # calculate the trivial lip
    trivial_Lip_sq *= np.linalg.norm(Wl)**2
    trivial_Lip = np.sqrt(trivial_Lip_sq)

    time_end = timeit.default_timer()
    # print(f'Time used = {time_end-time_begin}')
    return Lip_est, trivial_Lip, time_end-time_begin