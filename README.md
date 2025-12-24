# Transformer-based model for tabular data

> [!NOTE]
> Looking for the initial release (2023)? View the [initial release branch](https://github.com/wesselvanree/tabular-dl-transformer/tree/initial-release).

## Getting started

### Installing packages in a virtual environment

This project uses `uv` to manage its dependencies. [Install uv](https://docs.astral.sh/uv/getting-started/installation/) on your local machine. Clone this repository to your local machine, and open a terminal window in this repository. Then, install dependencies in a virtual environment using:

```
uv sync
```

### Data

Download the [Adult Data Set](https://archive.ics.uci.edu/ml/datasets/adult)
and store the files in `data/raw/adult`. This results in the following directory:

- `data/raw/adult`
  - `adult.data`
  - `adult.names`
  - `adult.test`

After installing the dependencies and the dataset, you can run one of the entrypoints:

- `hyperparams.py`: Run hyperparameter tuning, the type of model and encoding can be changed using command line
  arguments, run `python src/tabulardl/entrypoints/hyperparams.py --help` for more details.
- `train.py`: Train a model for different seeds using a given set of hyperparameters. The type of model, and other
  options can be altered using command line arguments. For more details, run `python src/tabulardl/entrypoints/train.py --help`.
