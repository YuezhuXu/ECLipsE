"""
Find good Lambda matrices for ECLipsE-Gen-Local algorithm
"""
import torch
import numpy as np
import cvxpy as cp
from numpy.linalg import eigh


def find_good_lambdas(Wi, Winext, Miprev, alphai, betai, algo):
    """
    Find optimal Lambda matrix for a layer using specified algorithm.
    
    Args:
        Wi: Weight matrix of current layer (di x diprev)
        Winext: Weight matrix of next layer (dinext x di)
        Miprev: M matrix from previous layer (diprev x diprev)
        alphai: Lower slope bounds for each neuron (di,)
        betai: Upper slope bounds for each neuron (di,)
        algo: Algorithm to use ('Acc', 'Fast', or 'CF')
    
    Returns:
        tuple: (Lambdai, ci, status, Xiprev, Mi)
            - Lambdai: Lambda matrix (di x di)
            - ci: Optimization objective value
            - status: 'Solved', 'Failed', or 'Skip'
            - Xiprev: X_i-1 matrix
            - Mi: M_i matrix
    """
    # Convert to appropriate types
    if isinstance(Wi, torch.Tensor):
        Wi = Wi.detach().cpu().numpy()
    if isinstance(Winext, torch.Tensor):
        Winext = Winext.detach().cpu().numpy()
    if isinstance(Miprev, torch.Tensor):
        Miprev = Miprev.detach().cpu().numpy()
    if isinstance(alphai, torch.Tensor):
        alphai = alphai.detach().cpu().numpy()
    if isinstance(betai, torch.Tensor):
        betai = betai.detach().cpu().numpy()
    
    alphai = np.asarray(alphai).flatten()
    betai = np.asarray(betai).flatten()
    
    diprev = Wi.shape[1]
    di = Wi.shape[0]
    
    Dalphai = np.diag(alphai)
    Dbetai = np.diag(betai)
    
    # Find active neurons (those with beta_i != alpha_i)
    active_idx = np.where(np.abs(betai - alphai) >= 1e-20)[0]
    fix_idx = np.where(np.abs(betai - alphai) < 1e-20)[0]
    di_active = len(active_idx)
    
    # If all neurons are affine (no active neurons), skip
    if len(active_idx) == 0:
        Lambdai = np.zeros((di, di))
        ci = 0
        status = 'Skip'
        Xiprev = None
        Mi = None
        return Lambdai, ci, status, Xiprev, Mi
    
    if algo == 'Acc':
        # Accurate algorithm: optimize each Lambda_i entry independently
        
        # Restrict matrices to active indices
        Wi_active = Wi[active_idx, :]
        Winext_active = Winext[:, active_idx]
        Dalphai_active = Dalphai[np.ix_(active_idx, active_idx)]
        Dbetai_active = Dbetai[np.ix_(active_idx, active_idx)]
        
        # Define CVX problem
        ci = cp.Variable()
        Li_gen_active = cp.Variable(di_active, nonneg=True)
        Lambdai_active = cp.diag(Li_gen_active)
        
        # Construct Schur complement matrix
        top_left = Lambdai_active - ci * (Winext_active.T @ Winext_active)
        top_right = 0.5 * Lambdai_active @ (Dalphai_active + Dbetai_active) @ Wi_active
        bottom_left = 0.5 * Wi_active.T @ (Dalphai_active + Dbetai_active) @ Lambdai_active
        bottom_right = Miprev + Wi_active.T @ Dalphai_active @ Lambdai_active @ Dbetai_active @ Wi_active
        
        Schur_X = cp.bmat([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        
        constraints = [
            Schur_X - 1e-15 * np.eye(Schur_X.shape[0]) >> 0,
            ci >= 0,
            Li_gen_active >= 0
        ]
        
        problem = cp.Problem(cp.Minimize(-ci), constraints)
        
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=2500, eps=1e-6)
        except:
            status = 'Failed'
            return np.eye(di), 0, status, None, None
        
        # Extract results
        Li_gen = np.zeros(di)
        if Li_gen_active.value is not None:
            Li_gen[active_idx] = Li_gen_active.value
            mean_active = np.mean(Li_gen_active.value)
            Li_gen[fix_idx] = min(1e20, 1e2 * mean_active)
        else:
            status = 'Failed'
            return np.eye(di), 0, status, None, None
        
        Lambdai = np.diag(Li_gen)
        ci_value = ci.value if ci.value is not None else 0
        
        # Compute Schur_X with actual values
        Schur_X_val = Schur_X.value if Schur_X.value is not None else None
        
        # Compute Xiprev and Mi
        Xiprev = Miprev + Wi.T @ Dalphai @ Lambdai @ Dbetai @ Wi
        Xiprev = (Xiprev + Xiprev.T) / 2 + 1e-30 * np.eye(diprev)
        
        Mi = Lambdai - 0.25 * Lambdai @ np.diag(alphai + betai) @ Wi @ np.linalg.pinv(Xiprev) @ Wi.T @ np.diag(alphai + betai) @ Lambdai
        Mi = (Mi + Mi.T) / 2 + 1e-30 * np.eye(di)
        
        # Check solution quality
        Schur_eig_min = np.min(np.linalg.eigvals(Schur_X_val)) if Schur_X_val is not None else -1
        Mi_eig_min = np.min(np.linalg.eigvals(Mi))
        
        if (Schur_eig_min > -1e-6) and (ci_value >= 1e-12) and np.all(Li_gen >= 0) and (Mi_eig_min > -1e-6):
            status = 'Solved'
        else:
            status = 'Failed'
        
        # Clamp for numerical stability
        Lambdai = np.minimum(1e20, Lambdai)
        
    elif algo == 'Fast':
        # Fast algorithm: use scalar lambda for all active neurons
        
        # Restrict matrices to active indices
        Wi_active = Wi[active_idx, :]
        Winext_active = Winext[:, active_idx]
        Dalphai_active = Dalphai[np.ix_(active_idx, active_idx)]
        Dbetai_active = Dbetai[np.ix_(active_idx, active_idx)]
        
        # Define CVX problem
        ci = cp.Variable()
        li_gen_active = cp.Variable(nonneg=True)
        Lambdai_active = li_gen_active * np.eye(di_active)
        
        # Construct Schur complement matrix
        top_left = Lambdai_active - ci * (Winext_active.T @ Winext_active)
        top_right = 0.5 * Lambdai_active @ (Dalphai_active + Dbetai_active) @ Wi_active
        bottom_left = 0.5 * Wi_active.T @ (Dalphai_active + Dbetai_active) @ Lambdai_active
        bottom_right = Miprev + Wi_active.T @ Dalphai_active @ Lambdai_active @ Dbetai_active @ Wi_active
        
        Schur_X = cp.bmat([
            [top_left, top_right],
            [bottom_left, bottom_right]
        ])
        
        constraints = [
            Schur_X - 1e-15 * np.eye(Schur_X.shape[0]) >> 0,
            ci >= 0,
            li_gen_active >= 0
        ]
        
        problem = cp.Problem(cp.Minimize(-ci), constraints)
        
        try:
            problem.solve(solver=cp.SCS, verbose=False, max_iters=2500, eps=1e-6)
        except:
            status = 'Failed'
            return np.eye(di), 0, status, None, None
        
        # Extract results
        li_value = li_gen_active.value if li_gen_active.value is not None else 0
        li_value = min(1e20, li_value)
        Lambdai = li_value * np.eye(di)
        ci_value = ci.value if ci.value is not None else 0
        
        # Compute Schur_X with actual values
        Schur_X_val = Schur_X.value if Schur_X.value is not None else None
        
        # Compute Xiprev and Mi
        Xiprev = Miprev + Wi.T @ Dalphai @ Lambdai @ Dbetai @ Wi
        Xiprev = (Xiprev + Xiprev.T) / 2 + 1e-30 * np.eye(diprev)
        
        Mi = Lambdai - 0.25 * Lambdai @ np.diag(alphai + betai) @ Wi @ np.linalg.pinv(Xiprev) @ Wi.T @ np.diag(alphai + betai) @ Lambdai
        Mi = (Mi + Mi.T) / 2 + 1e-30 * np.eye(di)
        
        # Check solution quality
        Schur_eig_min = np.min(np.linalg.eigvals(Schur_X_val)) if Schur_X_val is not None else -1
        Mi_eig_min = np.min(np.linalg.eigvals(Mi))
        
        if (Schur_eig_min > -1e-6) and (ci_value >= 1e-12) and (li_value >= 0) and (Mi_eig_min > -1e-6):
            status = 'Solved'
        else:
            status = 'Failed'
    
    elif algo == 'CF':
        # Closed-form algorithm (only works when alpha_i * beta_i >= 0)
        
        # Compute Lambda in closed form
        mat = (Dalphai + Dbetai) @ Wi @ np.linalg.pinv(Miprev) @ Wi.T @ (Dalphai + Dbetai)
        max_eig = np.max(np.linalg.eigvals(mat))
        Lambdai = (2.0 / max_eig) * np.eye(di)
        
        # Compute Xiprev and Mi
        Xiprev = Miprev + Wi.T @ Dalphai @ Lambdai @ Dbetai @ Wi
        Xiprev = (Xiprev + Xiprev.T) / 2 + 1e-30 * np.eye(diprev)
        
        Mi = Lambdai - 0.25 * Lambdai @ (Dalphai + Dbetai) @ Wi @ np.linalg.pinv(Xiprev) @ Wi.T @ (Dalphai + Dbetai) @ Lambdai
        Mi = (Mi + Mi.T) / 2 + 1e-30 * np.eye(di)
        
        # Compute ci
        fn = Winext @ np.linalg.pinv(Mi) @ Winext.T
        ci_value = 1.0 / np.max(np.linalg.eigvals(fn))
        
        status = 'Solved'
    
    else:
        raise ValueError(f'Invalid algorithm: {algo}')
    
    return Lambdai, ci_value, status, Xiprev, Mi
