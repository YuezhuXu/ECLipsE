import torch
import cvxpy as cp
import numpy as np
from numpy.linalg import eigh

def sdp(di, Wi_next, Ki_ep):
    # Wi_next and Ki_ep may be torch tensors or numpy arrays. Convert to numpy.
    if isinstance(Wi_next, torch.Tensor):
        Wi_next_np = Wi_next.detach().cpu().numpy()
    else:
        Wi_next_np = np.asarray(Wi_next)

    if isinstance(Ki_ep, torch.Tensor):
        Ki_ep_np = Ki_ep.detach().cpu().numpy()
    else:
        Ki_ep_np = np.asarray(Ki_ep)

    # Define variables
    s = cp.Variable()
    # represent diagonal Li by a nonnegative vector variable (faster / simpler)
    Li_diag = cp.Variable(di, nonneg=True)
    Li = cp.diag(Li_diag)

    # Compute constant matrices (numpy)
    Wi_next_T_Wi_next = Wi_next_np.T @ Wi_next_np

    # Use symmetric sqrt via eigendecomposition for speed and numerical stability
    # Ki_ep is symmetric; use eigh and clip small negative eigenvalues
    eigvals, eigvecs = eigh(Ki_ep_np)
    eigvals_clipped = np.clip(eigvals, a_min=0.0, a_max=None)
    sqrt_vals = np.sqrt(eigvals_clipped)
    sqrt_Ki_ep = (eigvecs * sqrt_vals) @ eigvecs.T

    # Form Schur complement matrix blocks (ensure numpy arrays for constants)
    top_left = Li - s * Wi_next_T_Wi_next
    top_right = Li @ sqrt_Ki_ep
    bottom_left = sqrt_Ki_ep @ Li
    bottom_right = np.eye(di, dtype=np.float64)

    Schur_X = cp.bmat([
        [top_left, top_right],
        [bottom_left, bottom_right]
    ])

    # Define problem
    constraints = [
        Schur_X >> 0,      # semidefinite
        s >= 1e-20,
        # ensure diagonal entries non-negative (already enforced by variable)
        # Li_diag >= 0  # variable declared nonneg
    ]

    problem = cp.Problem(cp.Minimize(-s), constraints)

    # Solve. Use SCS with tighter settings as a reasonable default; if you have
    # an interior-point SDP solver (CVXOPT, MOSEK), prefer that for speed/accuracy.
    problem.solve(solver=cp.SCS, verbose=False, max_iters=2500, eps=1e-6)

    # Access results
    s_value = float(s.value) if s.value is not None else None
    Li_value = Li.value
    # return Li as a torch tensor to match calling code
    Li_torch = torch.tensor(Li_value, dtype=torch.float64) if Li_value is not None else None
    return s_value, Li_torch, problem.status

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