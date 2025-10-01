import torch
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
    bottom_right = torch.eye(di, dtype=torch.float64)

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
    return s_value, torch.tensor(Li_value, dtype=torch.float64), problem.status

def ECLipsE(weights, alphas, betas):
    '''
        This function ...
            Args: ...
            Outputs: ...
    '''
    # length
    l = len(weights)
    
    trivial_Lip_sq = 1

    d0 = weights[0].shape[1]
    l0 = 0

    d_cum = 0
    Xi_prev = torch.eye(d0, dtype=torch.float64)

    for i in range(0, l-1):
        alpha, beta = alphas[i], betas[i]
        p = alpha * beta
        m = (alpha + beta) / 2
        
        di = weights[i].shape[0]
        Wi = weights[i]
        Wi_next = weights[i+1]

        Inv_Xi_prev = torch.linalg.inv(Xi_prev)

        Ki = m**2 * Wi @ Inv_Xi_prev @ Wi.T
        Ki = (Ki + Ki.T) / 2
        Ki_ep = Ki + (1e-10) * torch.eye(di, dtype=torch.float64)

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
        trivial_Lip_sq *= torch.linalg.norm(Wi)**2

    Wl = weights[l-1]
    eigvals, eigvecs = torch.linalg.eig(Wl.T @ Wl @ torch.linalg.inv(Xi))
    oneoverF = torch.max(eigvals.real)
    Lip_sq_est = oneoverF
    Lip_est = torch.sqrt(Lip_sq_est)

    return Lip_est