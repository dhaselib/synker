```markdown
# 📦 synker

**synker** is a Python package designed for generating synthetic datasets based on real data using kernel density estimation (KDE) methods. It supports bandwidth selection (Scott's and Silverman's rules), 2D KDE, synthetic data generation, Kullback–Leibler (KL) divergence evaluation between real and synthetic datasets, and arbitrary probability interval selection for data identification.

---

## 🚀 Features
- **Bandwidth Estimation**: Scott's Rule and Silverman's Rule  
- **2D Kernel Density Estimation**  
- **Synthetic Data Generation** based on KDE  
- **KL Divergence Calculation** for model evaluation  
- **Arbitrary Probability Interval Selection (pinkde)**: Identify data within specific probability ranges  
- Modular design for easy extension and usage  

---

## 🧠 Installation
Clone the repository and install the package locally:

```bash
git clone https://github.com/dhaselib/synker
cd synker
pip install .
```

---

## 🗂 Example Usage

```python
import numpy as np
import matplotlib.pyplot as plt
from synker.scott import Scott
from synker.silverman import Silverman
from synker.kl_div import KL_div
from synker.synthetic import Synthetic
from synker.kde import kde
from synker.kde import Pinkde
import pandas as pd

# Generate sample data
np.random.seed(42)
data = np.random.weibull(a=10, size=(1000, 2))

X = np.random.weibull(a=5, size=(1000))
Y = np.random.weibull(a=20, size=(1000))

# Test Silverman's bandwidth
hx = Silverman(X)
hy = Silverman(Y)

# Test Scott's bandwidth
hx = Scott(X)
hy = Scott(Y)

# KDE
syn_X = np.linspace(min(X), max(X), 100)
syn_Y = np.linspace(min(Y), max(Y), 100)
pkde = kde(X, Y, syn_X, syn_Y, hx, hy)
pkde = kde(X, Y, hx=hx, hy=hy, res=100)

# Synthetic data
synth_data = Synthetic(X=X, Y=Y, hx=hx, hy=hy, res=100)
synth_data = Synthetic(X, Y, bandwidth_method="Scott")
Synth_X = synth_data[:, 0]
Synth_Y = synth_data[:, 1]

# KL divergence
KL_divergence = KL_div(real_data=data, synthetic_data=synth_data, hx=hx, hy=hy)

# pinkde
grid_x = np.linspace(min(X), max(X), 100)
grid_y = np.linspace(min(Y), max(Y), 100)
res = 100
min_val = 0.2
max_val = 0.5
pinkde_result = pinkde(X, Y, hx, hy, "Scott", grid_x, grid_y, res, min_val, max_val)

# Plot
plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X, Y, label="Original Data")
plt.scatter(Synth_X, Synth_Y, label="Synthetic Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Original vs. Synthetic Data")

plt.figure(figsize=(8, 8), dpi=100)
plt.scatter(X, Y, label="Original Data", alpha=0.3)
plt.scatter(
    pinkde_result["X"],
    pinkde_result["Y"],
    label=f"P {min_val * 100}% , P {max_val * 100}%",
    color='red', alpha=0.3)
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.title("Pinkde Result")
plt.show()
```

---

## ✅ Testing
We provide a `test_synker.py` file to test core functionalities like Scott's Rule, KDE, synthetic sampling, KL divergence, and the new `pinkde` module:

### Run Tests:
```bash
python -m unittest test_synker.py
```

### Tests Include:
- Scott and Silverman bandwidth estimation  
- 2D KDE output validation  
- Synthetic data shape and bounds  
- Non-negative KL divergence check  
- Arbitrary probability interval selection using `pinkde`  

---

## 📁 Project Structure
```
synker/
│
├── __init__.py
├── scott.py
├── silverman.py
├── kde.py
├── sampling.py
├── kl_divergence.py
├── pinkde.py
├── tests/
│   └── test_synker.py
└── README.md
```

---

## 📜 License - MIT

```
MIT License

Copyright (c) 2025 Danial Haselibozchaloee

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
```"# 1" 
