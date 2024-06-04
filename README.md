# Transformer neural networks and quantum simulators: a hybrid approach for simulating strongly correlated systems

This is the implementation to our manuscript [Lange et al., arXiv:2406.00091](https://arxiv.org/abs/2406.00091). The Code contains parts that are adapted from [Zhang et al., PRB 107 (2023)](https://journals.aps.org/prb/abstract/10.1103/PhysRevB.107.075147) and [Sprague et al., Comm. Phys. 7 (2024)](https://www.nature.com/articles/s42005-024-01584-y).

Transformer quantum state implentation for the dipolar XY model, which can be pretrained using numerical data or experimental data, e.g. from quantum simulators as in Ref.  [Chen et al., Nature 616 (2023)](https://www.nature.com/articles/s41586-023-05859-2). Our hybrid training procedure consists of two parts: 

1. A data-driven pretraining: We train on snapshots from the computational basis as well as observables in different bases than the computational basis (here the X basis).
2. An energy-driven training using variational Monte Carlo.

<div align="center">
    <img width="479" alt="Momentum_git" src="https://github.com/HannahLange/HybridTransformer/blob/main/HybridTraining.jpg">
</div>

The source code is provided in the folder [src](https://github.com/HannahLange/HybridTransformer/tree/main/src). It contains `model.py` and `pos_encoding.py` with the implementation of the autoregressive, patched transformer that can be supplemented with spatial symmetries using `symmetries.py` as well as `localenergy.py` with the implementation of the Hamiltonian. Furthermore, we provide the exemplary run files `run_pretraining.py` and `run.py`.
