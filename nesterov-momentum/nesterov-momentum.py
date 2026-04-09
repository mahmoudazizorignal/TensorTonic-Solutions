import numpy as np

def nesterov_momentum_step(w, v, grad, lr=0.01, momentum=0.9):
    """
    Perform one Nesterov Momentum update step.
    """
    # Write code here
    w = np.array(w)
    v = np.array(v)
    grad = np.array(grad)

    v = momentum * v + lr * grad

    w = w - v

    return w, v