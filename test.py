import numpy as np
import matplotlib.pyplot as plt
from synker.scott import Scott
from synker.silverman import Silverman
from synker.kl_div import KL_div
from synker.synthetic import Synthetic
from synker.kde import kde
from synker.kde import Pinkde
import pandas as pd  # Import pandas

# Generate sample data
np.random.seed(42)
data = np.random.weibull(a=10, size=(1000, 2))

X = np.random.weibull(a=5, size=(1000))
Y = np.random.weibull(a=20, size=(1000))

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
pkde = kde(X, Y, hx=hx, hy=hy, res=100)

print("Probablity of KDE: \n", pkde)

# Test Synthetic
synth_data = Synthetic(X=X, Y=Y, hx=hx, hy=hy, res=100)
synth_data = Synthetic(X, Y, bandwidth_method="Scott")

Synth_X = synth_data[:, 0]
Synth_Y = synth_data[:, 1]
print("Synthetic Data: ", '\n', synth_data)

# Test KL_div
KL_divergence = KL_div(real_data=data, synthetic_data=synth_data, hx=hx, hy=hy)
print("KL divergence: \n", KL_divergence)

# Test pinkde
grid_x = np.linspace(min(X), max(X), 100)
grid_y = np.linspace(min(Y), max(Y), 100)
res = 100
min_val = 0.2
max_val = 0.5
pinkde_result = pinkde(X, Y, hx, hy, "Scott", grid_x, grid_y, res, min_val, max_val)
print("pinkde result:")
print(pinkde_result.head())  

# Plot
<<<<<<< HEAD
plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X, Y, label="Original Data")
plt.scatter(Synth_X, Synth_Y, label="Synthetic Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Original vs. Synthetic Data")

# Plot pinkde result
plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X, Y, label="Original Data", alpha=0.3)  #
plt.scatter(
    pinkde_result["X"],
    pinkde_result["Y"], label = f"P {min_val * 100}% , P {max_val * 100}% ",
    color = 'red', alpha = 0.3) 
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Pinkde Result")
plt.show()
=======
plt.figure(figsize = (8,8), dpi = 100)
plt.scatter(X,Y, label = "Real Data", alpha = 0.5)
plt.scatter(Synth_X,Synth_Y, label = "Synthetic Data", alpha = 0.5)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend(frameon = False, loc = 'lower left')
plt.show()
>>>>>>> 1440d6da110fa0d20c9770d5caab8240bcd9f02e
