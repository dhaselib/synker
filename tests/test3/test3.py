import numpy as np
import matplotlib.pyplot as plt
from synker.scott import Scott
from synker.silverman import Silverman
from synker.kl_div import KL_div
from synker.ckl_div import CKL_div
from synker.synthetic import Synthetic
from synker.kde import kde
from synker.pinkde import Pinkde
import pandas as pd

np.random.seed(42)
size_data = 2000
resolution = 50

X = np.random.beta(a=10,b = 10 , size = size_data)*20
Y = abs(np.random.gumbel(loc=1,scale = 5 , size = size_data))

data = np.column_stack((X, Y))

# Test Silverman's bandwidth
hx = Silverman(X)
hy = Silverman(Y)
print(f"Silverman's Bandwidth hx: {hx}, hy: {hy}")

# Test Scott's bandwidth
hx = Scott(X)
hy = Scott(Y)
print(f"Scott's Bandwidth hx: {hx}, hy: {hy}")

# Test kde
syn_X, syn_Y = np.linspace(min(X), max(X), 100), np.linspace(min(Y), max(Y), 100)
pkde = kde(X, Y, syn_X, syn_Y, hx, hy)
pkde = kde(X, Y, hx=hx, hy=hy, res=resolution)

print("Probablity of KDE: \n", pkde)

# Test Synthetic
synth_data = Synthetic(X=X, Y=Y, hx=hx, hy=hy, res=resolution)
synth_data = Synthetic(X, Y, bandwidth_method="Scott")

Synth_X = synth_data[:, 0]
Synth_Y = synth_data[:, 1]
print("Synthetic Data: ", '\n', synth_data)

# Test KL_div
KL_divergence = KL_div(real_2D_data=data, synthetic_2D_data=synth_data, hx=hx, hy=hy, res=resolution)
print("KL divergence: \n", KL_divergence)

# Test CKL_div
CKL_divergence = CKL_div(real_2D_data=data, synthetic_2D_data=synth_data, hx=None, hy=None, eps=1e-10, res=resolution)
print("CKL divergence (Real vs. Synthetic): \n", CKL_divergence)

# Test pinkde
grid_x = np.linspace(min(X), max(X), 100)
grid_y = np.linspace(min(Y), max(Y), 100)

min_val = 0.2
max_val = 0.5
pinkde_result = Pinkde(X, Y, hx, hy, "Scott", grid_x, grid_y, resolution, min_val, max_val)
print("pinkde result:")
print(pinkde_result.head()) 

# Plot
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(X, Y, alpha=0.6, label="Original Data")
plt.scatter(Synth_X, Synth_Y, alpha=0.6, label="Synthetic Data")
plt.xlabel("X", fontsize=18, fontfamily='serif')
plt.ylabel("Y", fontsize=18, fontfamily='serif')
plt.legend(frameon = False, fontsize = 14)
plt.tick_params(axis='both', labelsize=20)
plt.savefig(f"synthetic_comparison_{size_data}_res{resolution}.jpeg", dpi=600) 
plt.savefig(f"synthetic_comparison_{size_data}_res{resolution}.svg")



# Plot pinkde result
plt.figure(figsize=(8, 8), dpi=300)
plt.scatter(X, Y, label="Original Data", alpha=0.6)
plt.scatter(
    pinkde_result["X"],
    pinkde_result["Y"], label = f"P {min_val * 100}% , P {max_val * 100}% ",
    color = 'red', alpha = 0.3) 
plt.xlabel("X", fontsize=18, fontfamily='serif')
plt.ylabel("Y", fontsize=18, fontfamily='serif')
plt.legend(frameon = False, fontsize = 14)
plt.tick_params(axis='both', labelsize=20)
plt.show()
plt.savefig(f"pinkde_result_size_{size_data}_res{resolution}.jpeg", dpi=600)
plt.savefig(f"pinkde_result_size_{size_data}_res{resolution}.svg")

