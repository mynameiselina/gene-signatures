# A python package to provide tools for finding gene signatures in molecular data in respect to a specific phenotype.

## Tasks supported:
* feature selection
* feature extraction (not supported yet)
* classification
* regression (not supported yet)
* clustering (not supported yet)


## Applied in case studies of:
* Malignant Melanoma (not supported yet)
* Head and Neck squamous cell carcinoma
* Endometrial Carcinoma (not supported yet)


## Dependencies

- Python 3.6.3
- Numpy (>= 1.15)
- Pandas (>= 0.23)
- SciPy (>= 1.1.0)
- Scikit-Learn (>=0.19.2)
- Plac (>=1.0.0)
- Natsort (>=5.1.0)
- Matplotlib (>=2.2.3)
- Seaborn (>=0.9.0)
- Statsmodels (>=0.8.0)
- https://github.ibm.com/SysBio/plotting_utils (==1.0)
- https://github.ibm.com/SysBio/omics_processing (==1.0)

## Installation
Install `gene-signatures` after cloning:

```sh
pip3 install .
```
For developers is better to create an editable installation (symbolic links):

```sh
pip3 install -e .
```

## Example
Each script in the `gene-signatures` directory is an independent pipeline that can also be called from command line.

## Command line usage
```sh
command -config path_to_json_config/config.json [-D]
```
The available commands are the following:
```sh
set_up_data
process_data
rm_dpl_genes
nexus_express
feature_selection
combine_features
combine_cohorts
```

And examples of their corresponding config files can be found in the examples/configs directory.
Optionally the config parameters could also be set from command line. If the same parameter has been set in command line and in the config file then the one from the command line will be chosen.
The -D option is to set the debug mode on and have a more verbose printout.

## Guide for Head and Neck Squamous Cell Carcinoma (HNSCC) dataset
A combination of all the pipelines is proposed to analyze the HNSCC dataset and the syntax and order of the commands is written in the examples/configs/help_config_file.txt.

Moreover, for the final classification of the samples and the selection of the best model/signature follow the analysis from the  jupyter notebook in the examples/notebooks/Classification.ipynb