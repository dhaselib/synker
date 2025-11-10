import numpy as np
from .kde import kde

def KL_div(real_2D_data, synthetic_2D_data, hx, hy, grid=10, res=100, eps=1e-10):
    grid_x = np.linspace(min(real_2D_data[:, 0].min(), synthetic_2D_data[:, 0].min()), 
                          max(real_2D_data[:, 0].max(), synthetic_2D_data[:, 0].max()), 
                          grid)
    grid_y = np.linspace(min(real_2D_data[:, 1].min(), synthetic_2D_data[:, 1].min()), 
                          max(real_2D_data[:, 1].max(), synthetic_2D_data[:, 1].max()), 
                          grid)
    real_density = kde(real_2D_data[:, 0], real_2D_data[:, 1], grid_x, grid_y, hx, hy, res = res)
    synthetic_density = kde(synthetic_2D_data[:, 0], synthetic_2D_data[:, 1], grid_x, grid_y, hx, hy, res = res)
    real_density /= real_density.sum()
    synthetic_density /= synthetic_density.sum()
    real_density += eps
    synthetic_density += eps
    kl_divergence_value = np.sum(real_density * np.log(real_density / synthetic_density))

    return kl_divergence_value