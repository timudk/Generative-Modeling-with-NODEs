# Generative Modeling with Neural Ordinary Differential Equations

Code for reproducing the experiments of my thesis:
> to appear

The code is based on the code of the following paper:

> Will Grathwohl*, Ricky T. Q. Chen*, Jesse Bettencourt, Ilya Sutskever, David Duvenaud. "FFJORD: Free-form Continuous Dynamics for Scalable Reversible Generative Models." _International Conference on Learning Representations_ (2019).
> [[arxiv]](https://arxiv.org/abs/1810.01367) [[bibtex]](http://www.cs.toronto.edu/~rtqichen/bibtex/ffjord.bib)


## Prerequisites

Install `torchdiffeq` from https://github.com/rtqichen/torchdiffeq.

## Datasets

### Tabular (UCI + BSDS300)
Follow instructions from https://github.com/gpapamak/maf and place them in `data/`.

### VAE datasets
Follow instructions from https://github.com/riannevdberg/sylvester-flows and place them in `data/`.

<p align="center">
<img align="middle" src="./assets/github_flow.gif" width="400" height="400" />
</p>
