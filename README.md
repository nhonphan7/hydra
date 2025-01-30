# HYDRA: Symbolic feature engineering of overparameterized Eulerian hyperelasticity models for fast inference time

## Abstract

We introduce HYDRA, a learning algorithm that generates symbolic hyperelasticity models designed for running in 3D Eulerian hydrocodes that require fast and robust inference time. Classical deep learning methods require a large number of neurons to express a learned hyperelasticity model adequately. Large neural network models may lead to slower inference time when compared to handcrafted models expressed in symbolic forms. This expressivity-speed trade-off is not desirable for high-fidelity hydrocodes that require one inference per material point per time step. Pruning techniques may speed up inference by removing/deactivating less important neurons, but often at a non-negligible expense of expressivity and accuracy. In this work, we introduce a post-hoc procedure to convert a neural network model into a symbolic one to reduce inference time. Rather than directly confronting NP-hard symbolic regression in the ambient strain space, HYDRA leverages a data-driven projection to map strain onto a hyperplane and a neural additive model to parameterize the hyperplane via univariate bases. This setting enables us to convert the univariate bases into symbolic forms via genetic programming with explicit control of the expressivity-speed trade-off. Additionally, the availability of analytical models provides the benefits of ensuring the enforcement of physical constraints (e.g., material frame indifference, material symmetry, growth condition) and enabling symbolic differentiation that may further reduce the memory requirement of high-performance solvers. Benchmark numerical examples of material point simulations for shock loading in $\beta$-octahydro-1,3,5,7-tetranitro-1,3,5,7-tetrazocine ($\beta$-HMX) are performed to assess the practicality of using the discovered machine learning models for high-fidelity simulations.

## Overview of HYDRA

![alt text](https://github.com/nhonphan7/hydra/blob/main/figures/hydra.png)

Fig. 1: Two-step training process of HYDRA for an illustrative example of a 1D model in a 2D ambient strain space. (1) A linear transformation that projects strain onto a hyperplane and a set of nonlinear univariate bases are learned together using the neural additive model (NAM) in this projected feature space; (2) one symbolic regression (SR) is performed for each basis, where their sum constitutes the analytical form of the hyperelastic energy functional.

![alt text](https://github.com/nhonphan7/hydra/blob/main/figures/projected_nam.png)

Fig. 2: Inference step of a NAM in the projected feature space. A linear transformation (blue matrix) first projects the strain measure (red column) onto a hyperplane (purple column). Then, overparameterized neural additive branches independently map components of the projected measure into univariate bases. The final hyperelasticity model is a weighted sum of these feature functions in the mapped space.

![alt text](https://github.com/nhonphan7/hydra/blob/main/figures/expression_tree.png)

Fig. 3: Hierarchical structure of the expression tree. The learned hyperelastic energy functional is expressed as a linear combination of univariate bases, which further are functions of a linear combination of strain components.

Note that while HYDRA is set up to take the Eulerian finite strain and predict the free energy functional (and, in turn, the Eulerian analog of the second Piola-Kirchhoff stress), its framework can be modified for any inputs and outputs. For more information, please refer to our paper.

## Installation

### [Pip](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/)

Create custom environment:

```bash
python3 -m venv hydra
```

Activate environment:

```bash
source hydra/bin/activate
```

Upgrade pip:

```bash
python3 -m pip install --upgrade pip
python3 -m pip --version
```

Install [PyTorch](https://pytorch.org/) with CUDA (Linux):

```bash
python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu***
```

Replace `***` with a supported CUDA version.

### [Conda](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html)

Create custom environment:

```bash
conda create -n hydra python=3.8
```

Activate environment:

```bash
conda activate hydra
```

Install [PyTorch](https://pytorch.org/) with CUDA (Linux):

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=**.* -c pytorch -c nvidia
```

Replace `**.*` with a supported CUDA version.

### Both

Install requirements:

```bash
python3 -m pip install -r requirements.txt
```

Install [Juliaup](https://github.com/JuliaLang/juliaup) (for [PySR](https://github.com/MilesCranmer/PySR)):

```bash
curl -fsSL https://install.julialang.org | sh
```

## Create data sets

### Decoupled volumetric-isochoric setting of isotropic elasticity

```bash
cd isotropic
python3 stress_strain.py
```

### Third-order cubic elasticity for diamond crystals

```bash
cd cubic
python3 stress_strain.py
```

### Molecular dynamics simulations of $\beta$-HMX with monoclinic symmetry

```bash
cd md_PF
unzip all_Data_denoised_PF.csv.zip
cd ../md_eulerian
python3 stress_strain.py
```

## Neural additive models in the projected feature space

Run the following commands to use a saved model to evaluate the neural network bases for SR:

```bash
cd nam/data_set
python3 main.py
```

Replace `data_set` with `isotropic`, `cubic`, or `md_eulerian`. To train a new model, uncomment the lines `time = datetime.datetime.now().isoformat()` and `model.fit(X_train, y_train)` in `main.py` and comment out their subsequent line before running the commands above.

## Symbolic regression for neural network pruning

Run the following commands to train a symbolic model:

```bash
cd sr/data_set
python3 main.py
```

Replace `data_set` with `isotropic`, `cubic`, or `md_eulerian`.

## Citation

Please cite our paper if you find this code useful:

```bibtex
@article{phan2025hydra,
    title={HYDRA: Symbolic feature engineering of overparameterized Eulerian hyperelasticity models for fast inference time},
    author={Phan, Nhon N and Sun, WaiChing and Clayton, John D},
    journal={Computer Methods in Applied Mechanics and Engineering, in press},
    year={2025},
    publisher={Elsevier}
}
```
