# PINO for Semiconductor: AI-Powered Accelerated Semiconductor Device Simulation with NVIDIA PhysicsNeMo

> **Powered by NVIDIA PhysicsNeMo**
> 
> This project leverages the **NVIDIA PhysicsNeMo** framework to implement Physics-Informed Neural Operators (PINO) for high-speed, physics-preserving surrogate modeling of AI-powered semiconductor devices simulation. By utilizing NVIDIA's optimized operator implementations, we can achieve significant acceleration over traditional TCAD and Device Simulation tools while maintaining physical accuracy.

## Overview
The `PINO_for_Semiconductor` project aims to accelerate semiconductor device modeling by replacing computationally expensive Technology Computer-Aided Design (TCAD) simulations with Physics-Informed Neural Operators (PINO).

At the core of our approach is **NVIDIA PhysicsNeMo**, which provides the robust, differentiable, and GPU-accelerated spectral operators necessary to solve the complex PDEs governing device physics (Poisson, Continuity, and Charge Trap equations).

## Key Features

- **NVIDIA PhysicsNeMo Integration**:
  - Utilizes `physicsnemo.models.fno.FNO` for highly efficient spectral convolutions.
  - Leverages optimized FFT implementations for training on high-resolution physics fields.

- **Physics-Informed Learning (PINO)**:
  - Enforces the **Governing Physics** directly within the training loop.

- **High-Fidelity Surrogate Modeling**:
  - Specialized high-resolution modeling for critical gate oxide regions.

## Project Structure

### `code/`
Contains modular Python scripts for training and inference of PINO-based surrogate models for 3D SONOS NAND flash memory devices:

- **`sonos_dataset.py`**: Dataset loader for SONOS TCAD simulation data with normalization and preprocessing
- **`train_fno_model_sonos.py`**: Standard FNO training with MSE loss (data-driven only)
- **`train_pino_model_sonos.py`**: Physics-Informed Neural Operator (PINO) training with Sobolev loss (value + gradient matching)
- **`inference_time_sweep.py`**: Generate time-evolution animations of defect charge density
- **`inference_spacing_sweep.py`**: Parameter sweep for device geometry variations (thickness and spacing)
- **`optimize_sonos_lcm.py`**: Bayesian optimization to find optimal SONOS device configurations

### `results/`
Visualization outputs from trained models:

- **`sonos_time_evolution.gif`**: Time-evolution of trapped charge density
- **`spacing_sweep_10ks.gif`**: Defect density evolution across varying spacing parameters
- **`thickness_sweep_10ks.gif`**: Defect density evolution across varying thickness parameters  
- **`optimization_trends.png`**: Objective function convergence during Bayesian optimization
- **`optimization_cluster_3d.png`**: 3D visualization of optimal parameter clusters

### Data
Training data from Ginestra TCAD simulations is not included in this repository due to EULA/NDA restrictions.

## References
- **NVIDIA PhysicsNeMo**: [https://github.com/ NVIDIA/PhysicsNeMo](https://github.com/NVIDIA/PhysicsNeMo)
- Zongyi Li et al., Physics-Informed Neural Operator for Learning Partial Differential Equations, https://doi.org/10.48550/arXiv.2111.03794
- Gyujun Jeong et al., Physics-informed AI Surrogate Modeling for Accelerated Retention Prediction in Ferroelectric Vertical NAND, submitted