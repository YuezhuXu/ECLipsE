import timeit
import numpy as np
import cvxpy as cp
from scipy.linalg import sqrtm

def sdp(di, Wi_next, Ki_ep):
    # Wi_next, and Ki_ep are already defined numpy arrays
    # Wi_next: shape (n, di)
    # Ki_ep: shape (di, di)

    # Define variables
    s = cp.Variable()
    Li_gen = cp.Variable((di, 1))
    Li = cp.diag(cp.reshape(Li_gen, (di,), order='C'))  # reshape to 1D for diag

    # Compute constant matrices
    Wi_next_T_Wi_next = Wi_next.T @ Wi_next
    sqrt_Ki_ep = sqrtm(Ki_ep)  # convert to numpy array

    # Form Schur complement matrix
    top_left = Li - s * Wi_next_T_Wi_next
    top_right = Li @ sqrt_Ki_ep
    bottom_left = sqrt_Ki_ep @ Li
    bottom_right = np.eye(di)

    Schur_X = cp.bmat([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

    # Define problem
    constraints = [
        Schur_X >> 0,      # semidefinite
        s >= 1e-20,
        # Li >= 0
        Li_gen >= 0
    ]

    problem = cp.Problem(cp.Minimize(-s), constraints)

    # Solve
    problem.solve(solver=cp.SCS, verbose=False)

    # Access results
    s_value = s.value
    Li_value = Li.value
    return s_value, Li_value, problem.status

def eclipseE(weights):
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

    d0 = weights['w1'].shape[1]
    l0 = 0

    d_cum = 0
    Xi_prev = np.identity(d0)

    time_begin = timeit.default_timer()
    for i in range(1, l):
        di = weights['w'+str(i)].shape[0]
        Wi = weights['w'+str(i)]
        Wi_next = weights['w'+str(i+1)]

        Inv_Xi_prev = np.linalg.inv(Xi_prev)

        Ki = m**2 * Wi @ Inv_Xi_prev @ Wi.T
        Ki = (Ki + Ki.T) / 2
        Ki_ep = Ki + (1e-10) * np.identity(di)

        s_value, Li, status = sdp(di, Wi_next, Ki_ep)

        if status != cp.OPTIMAL:
            print('Problem status: ', status)
            break
        if s_value < 1e-20:
            print('Numerical issue')
            break

        Xi = Li - m**2 * Li @ Wi @ Inv_Xi_prev @ Wi.T @ Li
        Xi_prev = Xi
        d_cum = d_cum + di

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