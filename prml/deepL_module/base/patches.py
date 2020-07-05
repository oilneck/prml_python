import numpy as np

def ellipse2D_orbit(center:np.ndarray, cov:np.ndarray):
    assert cov.ndim == 2
    if center.ndim == 1: center = np.expand_dims(center, 1)
    D = np.linalg.cholesky(cov)
    theta = np.linspace(0, 2 * np.pi, 100)
    X = D @ np.array([np.cos(theta), np.sin(theta)])
    X += center
    return X.T
