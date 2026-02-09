# CrossLabFit

CrossLabFit is a framework for parameter estimation in dynamical systems that combines **quantitative data fitting** with **qualitative constraints (feasible windows)**.  
It provides an end-to-end pipeline to fit ODE models, explore parameter uncertainty, and analyze results with likelihood profiles and bootstrapping.  

The repository contains both **Python implementations** (for prototyping and visualization) and a **CUDA/C optimizer (qDE)** for efficient large-scale runs.  

---

## Repository Structure

```

├── crossLabFit.ipynb       # Main Jupyter notebook (standalone framework)
├── run_optimization.py     # Python module for optimization with DE
│
├── cudaDE                  # CUDA/C optimizer and simulations
│   ├── qDEcode_influenza      # CUDA/C source code for influenza model
│   ├── qDEcode_LV             # CUDA/C source code for LV models
│   └── simulations
│       ├── influenza               # Scripts & input files for influenza simulations
│       └── LVcycle                 # Scripts & input files for LV cycle simulations
│
├── data                    # Input data for the main notebook
│
└── miscellaneous           # Supplementary notebooks, tests, and analyses
    ├── cycle                   # Cycle Lotka–Volterra data
    ├── linear                  # Linear model data
    ├── 2-predators             # Two-predator Lotka–Volterra data
    ├── influenza               # Influenza model data
    ├── glycolysis              # Glycolysis model data
    ├── test                    # Testing different optimizers & cost functions data
    └── *.ipynb                 # Jupyter notebooks to analyze and plot results

````

---

## Usage

### 1. Python Framework
The core workflow is implemented in the **main Jupyter notebook**:

```bash
jupyter notebook crossLabFit.ipynb
````

This notebook walks through all steps of the pipeline:

* Data import
* Model definition
* Cost function setup with or without window constraints
* Differential Evolution optimization (`run_optimization.py`)
* Likelihood profiles
* Bootstrapping for parameter uncertainty
* Visualization and plots

The notebook uses:

* `run_optimization.py` for optimization routines
* Input data stored in the `data/` directory

---

### 2. CUDA/C Optimizer (qDE)

For large-scale simulations, a CUDA/C implementation of Differential Evolution (`qDE`) is provided.

To build:

```bash
cd cudaDE/qDEcode_influenza   # or cudaDE/qDEcode_LV
make
```

To run bootstrap or likelihood profile simulations:

```bash
cd cudaDE/simulations/influenza   # or LVcycle
zsh bootstrap.sh
zsh likelihood.sh
```

Outputs are stored in the corresponding simulation folders.

---

### 3. Miscellaneous Analyses

The `miscellaneous/` folder contains:

* Extra Jupyter notebooks for visualizations, testing alternative optimizers, and window constraint construction
* Subfolders with generated data and profiles for each case study (`cycle`, `linear`, `2-predators`, `influenza`)
* `test/` with experiments from development

---

## Case Studies

CrossLabFit includes example models:

* **Cycle Lotka–Volterra**
* **Two-predator Lotka–Volterra**
* **Linear system**
* **Influenza virus model (with T cell constraints)**

Each study has corresponding simulation outputs and analysis notebooks.

---

## Requirements

Python environment:

* `numpy`, `scipy`, `matplotlib`, `seaborn`, `pandas`
* `jupyter` for notebooks

CUDA build:

* Linux environment with `nvcc`

---

## Citation

If you use **CrossLabFit** in your work, please cite:

Blanco-Rodriguez, R., Miura, T. A., & Hernandez-Vargas, E. (2025). CrossLabFit: A novel framework for integrating qualitative and quantitative data across multiple labs for model calibration. PLOS Computational Biology, 21(11), e1013704. https://doi.org/10.1371/journal.pcbi.1013704
