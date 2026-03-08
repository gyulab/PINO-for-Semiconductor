# PINO for Semiconductor: AI-Powered Accelerated Semiconductor Device Simulation with NVIDIA PhysicsNeMo

> **Powered by NVIDIA PhysicsNeMo**
> 
> This project leverages the **NVIDIA PhysicsNeMo** framework to implement Physics-Informed Neural Operators (PINO) for high-speed, physics-preserving surrogate modeling of AI-powered semiconductor devices simulation. By utilizing NVIDIA's optimized operator implementations, we can achieve significant acceleration over traditional TCAD and Device Simulation tools while maintaining physical accuracy.

## Overview
[![Watch the Tutorial Video](https://youtu.be/hUIz8ZxcY7w/maxresdefault.jpg)](https://youtu.be/hUIz8ZxcY7w)
^ Watch the Tutorial Video!

The `PINO_for_Semiconductor` project aims to accelerate semiconductor device modeling by replacing computationally expensive Technology Computer-Aided Design (TCAD) simulations with Physics-Informed Neural Operators (PINO).

At the core of our approach is **NVIDIA PhysicsNeMo**, which provides the robust, differentiable, and GPU-accelerated spectral operators necessary to solve the complex PDEs governing device physics (Poisson, Continuity, and Charge Trap equations).

## Key Features

- **NVIDIA PhysicsNeMo Integration**:
  - Utilizes NVIDIA's `physicsnemo` library for highly efficient spectral convolutions.
  - Leverages optimized FFT implementations for training on high-resolution physics fields.

- **Physics-Informed Learning (PINO)**:
  - Enforces the **Governing Physics** directly within the training loop, e.g., Poisson, Continuity, and Charge Trap equations.

- **High-Fidelity Surrogate Modeling**:
  - Specialized high-resolution modeling for critical gate oxide regions.

## Project Structure

### Documentation
- `README.md`: Project overview and setup instructions.
- `docs/`: Additional documentation and resources.
- `AI-Surrogate-modeling-tutorial.pdf`: Tutorial for AI Surrogate Modeling.

### `code/`
Contains modular Python scripts for training and inference of PINO-based surrogate models for 3D SONOS NAND flash memory devices:

- **`train_fno_model_sonos.py`**: Standard FNO training with MSE loss (data-driven only)
- **`train_pino_model_sonos.py`**: Physics-Informed Neural Operator (PINO) training with Poisson Loss
- **`inference_time_sweep.py`**: Generate time-evolution animations of defect charge density

### `results/`
Visualization outputs from trained models:
- `fno_sonos_time.gif`: FNO (Data-driven) Time-evolution of trapped charge density
- `fno_sonos_thickness.gif`: FNO (Data-driven) Thickness evolution of trapped charge density
- `pino_sonos_time.gif`: PINO (Physics-Informed) Time-evolution of trapped charge density
- `pino_sonos_thickness.gif`: PINO (Physics-Informed) Thickness evolution of trapped charge density

### Data
Training data from AMAT Ginestra simulations is not included in this repository due to EULA/NDA restrictions.

## References
- **NVIDIA PhysicsNeMo**: [https://github.com/ NVIDIA/PhysicsNeMo](https://github.com/NVIDIA/PhysicsNeMo)
- Zongyi Li et al., Physics-Informed Neural Operator for Learning Partial Differential Equations, https://doi.org/10.48550/arXiv.2111.03794.
