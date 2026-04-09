import numpy as np

def adadelta_step(w, grad, E_grad_sq, E_update_sq, rho=0.9, eps=1e-6):
    """
    Perform one AdaDelta update step.
    """
    # Write code here
    w = np.array(w)
    grad = np.array(grad)
    E_grad_sq = np.array(E_grad_sq)
    E_update_sq = np.array(E_update_sq)

    E_grad_sq = rho * E_grad_sq + (1 - rho) * grad**2
    update_size = -np.sqrt(E_update_sq + eps) / np.sqrt(E_grad_sq + eps) * grad
    E_update_sq = rho * E_update_sq + (1 - rho) * update_size**2

    w = w + update_size

    return w, E_grad_sq, E_update_sq