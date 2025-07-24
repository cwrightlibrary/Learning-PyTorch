# Mathematics for Machine Learning

This will function as the table of contents for my notebook workthroughs of the content.

## Table of Contents

- [Tables](mathematics-for-machine-learning/0-tables.ipynb)
- [**1. Mathematical foundations**](mathematics-for-machine-learning/1-mathematical-foundations.ipynb)
- [**2. Linear algebra**](mathematics-for-machine-learning\2-linear-algebra.ipynb)

---

## To-do Python practice

- **2. Linear algebra basics**
  - [ ] **Implement vector and matrix operations**: dot product, matrix multiplication, transpose, inverse.
  - [ ] **Visualize transformations**: Plot vectors before and after applying 2×2 matrices.
  - [ ] **Solve linear systems**: Use NumPy to solve $Ax=b$ and compare with your own Gaussian elimination.
- **3. Analytic geometry**
  - [ ] Write functions to compute distances between points, point-to-line distance.
  - [ ] Create code to check whether a point lies inside a polygon.
- **4. Matrix decompositions**
  - [ ] Use `numpy.linalg` to compute eigenvalues/eigenvectors and SVD.
  - [ ] Apply PCA to a simple dataset and visualize variance explained.
- **5. Vector calculus**
  - [ ] Implement gradients of scalar functions via finite differences; verify using `autograd` or `jax`.
  - [ ] Visualize gradient fields in 2D using quiver plots.
- **6. Probability and distributions**
  - [ ] Code PMF/PDF for common distributions (Normal, Bernoulli).
  - [ ] Generate samples via inversion/accept-reject methods, plot histograms and compare to theoretical PDF.
- **7. Probabilistic modeling**
  - [ ] Build a Bayesian coin-flip model: update posterior Beta distribution after observing flips.
  - [ ] Simulate posterior predictive distribution and visualize credible intervals.
- **8. Continuous case: maximization, gradient descent**
  - [ ] Implement gradient descent to minimize a univariate quadratic; then extend to linear regression.
  - [ ] Track convergence and experiment with learning rates.
- **9. Modeling with linear regression**
  - [ ] Build simple and ridge regression using matrix formulas; compare with `scikit-learn`.
  - [ ] Explore bias-variance tradeoff via bootstrap experiments.
- **10. Classification and logistic regression**
  - [ ] Implement logistic regression (batch and SGD); report accuracy on a toy dataset.
  - [ ] Visualize decision boundary and its update during training.
- **11. Generative models**
  - [ ] Fit Gaussian discriminant analysis (GDA) classifier and compare with logistic regression.
  - [ ] Draw contour plots of class-conditional Gaussians.
- **12. Bayesian linear regression**
  - [ ] Derive posterior mean and covariance; code predictive distribution in Python.
  - [ ] Visualize predictive mean ± uncertainty bands.
- **13. Bayesian logistic regression**
  - [ ] Implement Laplace approximation for logistic regression posterior.
  - [ ] Plot approximate posterior and compare to point estimates.
- **14. Kernel methods**
  - [ ] Implement basic kernels (RBF, polynomial), and apply kernel ridge regression on 1D data.
  - [ ] Compare feature-space vs kernel-space solutions.
- **15. Gaussian procceses (GPs)**
  - [ ] Code a GP regressor with RBF kernel; sample from prior/posterior.
  - [ ] Optimize hyperparameters (length-scale, variance) via log-marginal likelihood.
- **16. Kernelization vs Bayesian regression**
  - [ ] Compare kernel ridge regression with GP regression on same dataset.
  - [ ] Visualize predictive variance and discuss differences.

---

## Resources for projects

### 🧮 Math & Linear Algebra

| Library        | Use Case                                                                   |
| -------------- | -------------------------------------------------------------------------- |
| `numpy`        | Core numerical computing (vectors, matrices, dot products, linear systems) |
| `scipy.linalg` | Advanced matrix decompositions (eigen, SVD, inverse, QR, etc.)             |
| `sympy`        | Symbolic math (derivatives, solving equations, algebraic simplification)   |

---

### 📊 Visualization

| Library      | Use Case                                                      |
| ------------ | ------------------------------------------------------------- |
| `matplotlib` | Plotting vectors, functions, surfaces, 2D/3D visuals          |
| `seaborn`    | Statistical visualization with clean defaults                 |
| `plotly`     | Interactive 3D plots (useful for visualizing surfaces or GPs) |
| `ipywidgets` | Add sliders/buttons to Jupyter for interactive demos          |

---

### 📈 Statistics & Distributions

| Library       | Use Case                                                             |
| ------------- | -------------------------------------------------------------------- |
| `scipy.stats` | Probability distributions (PDF, CDF, sampling, fitting)              |
| `statsmodels` | Statistical modeling (linear regression, confidence intervals, etc.) |

---

### 🤖 Machine Learning

| Library            | Use Case                                                  |
| ------------------ | --------------------------------------------------------- |
| `scikit-learn`     | Baseline models for regression, classification, PCA, etc. |
| `pymc` or `pystan` | Probabilistic programming for Bayesian inference          |
| `gpytorch`         | Gaussian Process modeling in PyTorch                      |

---

### 🔁 Optimization & Autograd

| Library          | Use Case                                                               |
| ---------------- | ---------------------------------------------------------------------- |
| `jax`            | Automatic differentiation (autograd), JIT compilation, vectorized math |
| `autograd`       | Simpler alternative to JAX for differentiating Python code             |
| `scipy.optimize` | Solving minimization problems, fitting models                          |

---

### 📦 Optional Extras (for advanced users)

| Library        | Use Case                                                             |
| -------------- | -------------------------------------------------------------------- |
| `torch`        | Deep learning + autodiff with full control                           |
| `einops`       | Elegant tensor reshaping and composition, useful for higher-dim math |
| `linearmodels` | Advanced regression tools (e.g., Bayesian linear models)             |

---

### 🧪 Development Environment

| Tool           | Use Case                              |
| -------------- | ------------------------------------- |
| `JupyterLab`   | Interactive coding and markdown notes |
| `VSCode`       | IDE for Python with Jupyter support   |
| `Google Colab` | Run Jupyter notebooks in the cloud    |
