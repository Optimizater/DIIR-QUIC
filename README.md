# DIIR-QUIC: Sparse Inverse Covariance Estimation with Nonconvex Regularization

## Introduction

DIIR-QUIC is a novel **Damped Inexact Iteratively Reweighted** algorithm based on the **QUadratic approximation for sparse Inverse Covariance (QUIC)** method. It solves large-scale sparse inverse covariance (precision) matrix estimation problems with **nonconvex partly smooth regularizers**.

### Problem Formulation

The algorithm addresses the following regularized log-determinant optimization problem:

$$
\min_{\boldsymbol{X} \succ 0} \left\{ \text{tr}(\boldsymbol{S}\boldsymbol{X}) - \log \det \boldsymbol{X} + \rho \sum_{i,j} \phi(|X_{ij}|) \right\},
$$

where:

- $\boldsymbol{X}$ is the precision matrix (inverse covariance) to be estimated.
- $\boldsymbol{S}$ is the empirical covariance matrix.
- $\phi(\cdot)$ is a **nonconvex sparsity-promoting regularizer** (e.g., $\ell_p$-norm, SCAD, MCP).
- $\rho > 0$ is a regularization parameter.

## Installation

### Prerequisites

- Python 3.10 or later.
- C++ compiler (for building the core module).

### Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/Optimizater/DIIR-QUIC.git
   cd DIIR-QUIC
   ```

2. Install Python dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Build the core C++ module (follow instructions in [`/diir_quic/core/README.md`](/diir_quic/core/README.md)).

## Example

Run the provided example script [`test.py`](test.py):

```bash
python test.py
```

This script:

1. Generates a synthetic tridiagonal precision matrix.
2. Estimates the matrix using DIIR-QUIC with an $\ell_p$-norm regularizer.

## References

1. Original Paper:  
   _Efficient QUIC-Based Damped Inexact Iterative Reweighting for Sparse Inverse Covariance Estimation with Nonconvex Partly Smooth Regularization_ ([PDF](https://optimization-online.org/?p=30821)).
