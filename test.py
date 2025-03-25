import numpy as np
from synker.scott import Scott
from synker.silverman import Silverman
from synker.kde import KDE_2D
from synker.kl_div import KL_div
from synker.synthetic import Synthetic


# Generate sample data
np.random.seed(42)
data = np.random.weibull(a=2, size=(100, 2))

# Test Scott's bandwidth
hx = Scott(data[:, 0])
hy = Scott(data[:, 1])
print(f"Scott's Bandwidth hx: {hx}, hy: {hy}")

# Test Silverman's bandwidth
hx_sil = Silverman(data[:, 0])
hy_sil = Silverman(data[:, 1])
print(f"Silverman's Bandwidth hx: {hx_sil}, hy: {hy_sil}")

# Test KDE
grid_x = np.linspace(data[:, 0].min(), data[:, 0].max(), 100)
grid_y = np.linspace(data[:, 1].min(), data[:, 1].max(), 100)
density = KDE_2D(data[:, 0], data[:, 1], grid_x, grid_y, hx, hy)
print(f"KDE Density Shape: {density.shape}")

# Test Synthetic
Synth_data = Synthetic(data, hx=hx, hy=hy, grid_x=grid_x, grid_y=grid_y, n_samples=100)
print(f"Synthetic Data Shape: {Synth_data.shape}")

# Test KL Divergence 
kl = KL_div(data, Synth_data, hx, hy)
print(f"KL Divergence (self): {kl}")
