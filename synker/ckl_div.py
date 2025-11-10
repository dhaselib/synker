import numpy as np
from .kde import kde

def cdf_kde(x, y, xi=None, yi=None, hx=None, hy=None, res=100):
   
    n = len(x)
    
    if xi is None:
        min_x, max_x = np.min(x), np.max(x)
    
        xi = np.linspace(min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1, res)
    if yi is None:
        min_y, max_y = np.min(y), np.max(y)
        yi = np.linspace(min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1, res)

    
    if hx is None:
        hx = 1.06 * np.std(x) * n**(-1/5)
    if hy is None:
        hy = 1.06 * np.std(y) * n**(-1/5)

    
    if hx == 0:
        hx = 1e-5 
    if hy == 0:
        hy = 1e-5 

    pdf_grid = np.zeros((len(xi), len(yi)))

    for i_data in range(n):

        kernel_x = (1 / (hx * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x[i_data] - xi) / hx)**2)
        kernel_y = (1 / (hy * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((y[i_data] - yi) / hy)**2)

        pdf_grid += np.outer(kernel_x, kernel_y)

    pdf_grid /= n

    dx = xi[1] - xi[0]
    dy = yi[1] - yi[0]

    pdf_mass = pdf_grid * dx * dy

    cdf_grid = np.cumsum(np.cumsum(pdf_mass, axis=0), axis=1)
    cdf_grid = np.clip(cdf_grid, 0.0, 1.0)

    return cdf_grid

import numpy as np


def CKL_div(real_2D_data, synthetic_2D_data, hx=None, hy=None, eps=1e-10, grid_size=None, res=100):

    if grid_size is None:
        num_points = res
    else:

        num_points = grid_size 

    min_x = min(real_2D_data[:, 0].min(), synthetic_2D_data[:, 0].min())
    max_x = max(real_2D_data[:, 0].max(), synthetic_2D_data[:, 0].max())
    min_y = min(real_2D_data[:, 1].min(), synthetic_2D_data[:, 1].min())
    max_y = max(real_2D_data[:, 1].max(), synthetic_2D_data[:, 1].max())

    grid_x = np.linspace(min_x - (max_x - min_x) * 0.1, max_x + (max_x - min_x) * 0.1, num_points)
    grid_y = np.linspace(min_y - (max_y - min_y) * 0.1, max_y + (max_y - min_y) * 0.1, num_points)

    real_cdf = cdf_kde(real_2D_data[:, 0], real_2D_data[:, 1], grid_x, grid_y, hx, hy, res)
    synthetic_cdf = cdf_kde(synthetic_2D_data[:, 0], synthetic_2D_data[:, 1], grid_x, grid_y, hx, hy, res)

    F_clipped = np.clip(real_cdf, eps, 1 - eps)
    G_clipped = np.clip(synthetic_cdf, eps, 1 - eps)

    ckl_value = np.sum(F_clipped * np.log(F_clipped / G_clipped) - (F_clipped - G_clipped))
    ckl_value += np.sum((1 - F_clipped) * np.log((1 - F_clipped) / (1 - G_clipped)) - ((1 - F_clipped) - (1 - G_clipped)))

    return ckl_value