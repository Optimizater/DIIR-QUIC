# Introduction

DIIR-QUIC[^ref] is a novel Damped Inexact Iteratively Reweighted algorithm based on the QUadratic approximation for sparse Inverse Covariance (QUIC) method.

# Installation

You need to build the core module according to [instruction](/diir_quic/core/README.md).

> [!NOTE] > **NOTE:** The code is written using Python 3.10.12 and it is recommended to install the dependencies in [requirements.txt](requirements.txt)

You can use [test.py](test.py) as an example to understand how to use DIIR-QUIC to estimate the precision matrix of a random tridiagonal matrix.

[^ref]: _Eﬀicient QUIC-Based Damped Inexact Iterative Reweighting for Sparse Inverse Covariance Estimation with Nonconvex Partly Smooth Regularization_
