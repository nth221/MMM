<img width="930" height="320" alt="MMM_logo" src="https://github.com/user-attachments/assets/b70d9ffc-5191-46a6-b073-d130d305a42e" />


# MMM: Quantum-Chemical Molecular Representation Learning for Combinatorial Drug Recommendation
## This repository contains the official implementation of **"MMM: Quantum-Chemical Molecular Representation Learning for Combinatorial Drug Recommendation"**.

## Experimental Setup

We provide the network architecture of the proposed MMM model, along with the pipeline code to enable users to train and test the network on the EHR dataset and drug molecular ELF image. The DDI calculations and drug information used in this work are based on the implementation from [SafeDrug](https://github.com/ycq091044/SafeDrug)'s repository. All experiments were conducted in an environment with Python 3.9.23, PyTorch 2.3.0+cu118, and CUDA 11.8.

### Citation
If you find this code useful for your work, please cite the following and consider starring this repository:
```
@inproceedings{
kwon2025mmm,
title={{MMM}: Quantum-Chemical Molecular Representation Learning for Personalized Drug Recommendation},
author={Chongmyung Kwon and Yujin Kim and Seoeun Park and Yunji Lee and Charmgil Hong},
booktitle={PRedictive Intelligence in MEdicine},
year={2025},
organization={Springer}
}
```