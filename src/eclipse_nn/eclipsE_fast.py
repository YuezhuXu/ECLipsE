import torch

def ECLipsE_Fast(weights, alphas, betas):
    '''
        This function ...
            Args: ...
            Outputs: ...
    '''
    # length
    l = len(weights)

    for i in range(0, l-1):
        alpha, beta = alphas[i], betas[i]
        p = alpha * beta
        m = (alpha + beta) / 2

        di = weights[i].shape[0]
        Wi = weights[i]

        Xi_prev = Xi if i > 1 else torch.eye(weights[i].shape[1], dtype=torch.float64)
        Inv_Xi_prev = torch.linalg.inv(Xi_prev)

        mat = Wi @ Inv_Xi_prev @ Wi.T
        eigvals, eigvecs = torch.linalg.eig(mat)
        li = 1 / (2 * m**2 * torch.max(eigvals.real))
        Xi = li * torch.eye(di, dtype=torch.float64) - li**2 * m**2 * mat


    Wl = weights[l-1]
    eigvals, eigvecs = torch.linalg.eig(Wl.T @ Wl @ torch.linalg.inv(Xi))
    oneoverF = torch.max(eigvals.real)
    Lip_sq_est = oneoverF
    Lip_est = torch.sqrt(Lip_sq_est)
    
    return Lip_est