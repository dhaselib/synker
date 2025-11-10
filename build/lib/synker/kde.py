import numpy as np

def kde(x, y, grid_x=None, grid_y=None, hx=None, hy=None, res=100):
    
    n = len(x)
    
    if grid_x is None:
        grid_x = np.linspace(min(x), max(x), res)
    if grid_y is None:
        grid_y = np.linspace(min(y), max(y), res)

    p = np.zeros((len(grid_x), len(grid_y)))

    for i in range(n):
        p1 = np.exp(-((x[i] - grid_x) ** 2) / (2 * hx ** 2))
        p2 = np.exp(-((y[i] - grid_y) ** 2) / (2 * hy ** 2))
        p += (1 / (n * hx * hy)) * p1[:, np.newaxis] * p2[np.newaxis, :]

    return p