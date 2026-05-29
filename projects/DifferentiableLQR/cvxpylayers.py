import cvxpy as cp
import torch
from cvxpylayers.torch import CvxpyLayer

def CvxpyLayersSolver(Q, p, G, H):
    '''
        Solve QP: 
            min 1/2 z^T Q z + z^T p
            s.t. G z <= H
        Q: [B, nz, nz]
        p: [B, nz]
        G: [B, n_ineq, nz]
        H: [B, n_ineq]
    '''
    B, nz = Q.shape[0], Q.shape[1]
    n_ineq = G.shape[1]
    z = cp.Variable(nz)
    Q_param = cp.Parameter((nz, nz), PSD=True)
    P_param = cp.Parameter(nz)
    G_param = cp.Parameter((n_ineq, nz))
    H_param = cp.Parameter(n_ineq)
    objective = cp.Minimize(0.5 * cp.quad_form(z, Q_param) + P_param.T @ z)
    # constraints = [G_param @ z <= H_param]
    constraints = [G_param @ z - H_param <= 0]
    problem = cp.Problem(objective, constraints)
    # assert problem.is_dpp()
    cvxlayer = CvxpyLayer(problem, parameters=[Q_param, P_param, G_param, H_param], variables=[z])
    
    sol = torch.zeros_like(p)
    for i in range(B):
        sol_i = cvxlayer(Q[i], p[i], G[i], H[i])
        sol[i] = sol_i
    return sol

    objective = cp.Minimize(0.5 * z.T @ Q_param @ z + P_param.T @ z)