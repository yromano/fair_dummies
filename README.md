# Fair Dummies: Achieving Equalized Odds by Resampling Sensitive Attributes

This package implements "Fair Dummies": a flexible framework [1] for learning predictive models that approximately satisfy the equalized odds notion of fairness. This is achieved by introducing a general discrepancy function that rigorously quantifies violations of this criterion, formulating a differentiable penalty that drives the model parameters towards equalized odds.

To rigorously evaluate fitted models, we also implement a formal hypothesis test to detect when a prediction rule violates the equalized odds property. Both the model fitting and hypothesis testing leverage a resampled version of the sensitive attribute obeying the equalized odds property by construction.

Lastly, we demonstrate how to incorporate techniques for equitable uncertainty quantification---unbiased for each protected group---to precisely communicate the results of the data analysis.

[1] Y. Romano, S. Bates, and E. J. Candès, “Achieving Equalized Odds by Resampling Sensitive Attributes.” 2020.

[2] Y. Romano, E. Patterson, and E. J. Candès, [“Conformalized quantile regression.”](https://arxiv.org/abs/1905.03222) 2019.

[3] Y. Romano, R. F. Barber, C. Sabbatti and E. J. Candès, [“With malice towards none: Assessing uncertainty via equalized coverage.”](https://statweb.stanford.edu/~candes/papers/EqualizedCoverage.pdf) 2019.

## Getting Started

The implementation of [1] is self-contained and written in python.

Part of the code is a taken from:
* Conformalized quantile regression (CQR) [2] and equalized coverage [3] frameworks for constructing distribusion-free prediction intervals/sets. Code is avaialable at https://github.com/yromano/cqr
* nonconformist package available at https://github.com/donlnz/nonconformist
* We compare Fair Dummies to Adversarial Debiasing [4]; our implementation is based on https://github.com/equialgo/fairness-in-ml
* We compare Fair Dummies to HGR [5], where our code is based on https://github.com/equialgo/fairness-in-ml

### Prerequisites

* python
* numpy
* scipy
* scikit-learn
* scikit-garden
* pytorch
* pandas

### Installing

The development version is available here on github:
```bash
git clone https://github.com/yromano/fair_dummies.git
```

## Usage

### CQR

Please refer to [synthetic_experiment.ipynb](synthetic_experiment.ipynb) for basic usage. Comparisons to competitive methods and additional usage examples of this package can be found in [all_classification_experiments.py](all_classification_experiments.py) and [all_regression_experiments.py](all_regression_experiments.py).

## Reproducible Research

The code available under synthetic_experiment.ipynb,  all_classification_experiments.py, and all_regression_experiments.py in the repository replicates all experimental results in [1].

### Publicly Available Datasets

* [Communities and Crimes](http://archive.ics.uci.edu/ml/datasets/communities+and+crime): UCI Communities   and   crime   data   set.

* [Nursery](https://archive.ics.uci.edu/ml/datasets/nursery): UCI Nursery data   set.

### Data subject to copyright/usage rules

The Medical Expenditure Panel Survey (MPES) data can be downloaded by following [this explanation](https://github.com/yromano/cqr/blob/master/get_meps_data/README.md) (code provided by [IBM's AIF360](https://github.com/IBM/AIF360)).

* [MEPS_21](https://meps.ahrq.gov/mepsweb/data_stats/download_data_files_detail.jsp?cboPufNumber=HC-192): Medical expenditure panel survey,  panel 21.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
