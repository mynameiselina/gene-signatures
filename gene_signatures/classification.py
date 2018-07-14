# custom imports
from omics_processing.io import (
    set_directory, load_clinical
)
from omics_processing.remove_duplicates import (
    remove_andSave_duplicates
)
from gene_signatures.core import (
    custom_div_cmap,
    get_chr_ticks,
    choose_samples
)

# basic imports
import os
import numpy as np
import pandas as pd
import json
from scipy.spatial.distance import pdist, squareform
from natsort import natsorted, index_natsorted
import math
import logging
from sklearn import linear_model

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

script_fname = os.path.basename(__file__).rsplit('.')[0]
script_path = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def _run_classification(dat, dat_target):
    model = linear_model.LogisticRegression(penalty = 'l1', C = 1)
    estimators = []
    correct = 0
    wrong = 0
    all_coefs = np.zeros(dat.shape)
    for choose_sample in range(dat.shape[0]):
        X = dat.drop(dat.index[choose_sample:choose_sample+1])
        y = dat_target.drop(dat.index[choose_sample:choose_sample+1])
        one_sample = dat.iloc[choose_sample:choose_sample+1,:]
        y_real = dat_target.iloc[choose_sample:choose_sample+1]

        model.fit(X, y)

        # plt.plot(model.coef_[0])
        all_coefs[choose_sample:choose_sample+1, :] = model.coef_[0]
        # plt.scatter(range(X.shape[1]),model.coef_[0], color='k')

        y_pred = model.predict(one_sample)

        if (y_pred[0] == y_real.values[0]):
            correct = correct + 1
        else:
            wrong = wrong + 1


    return all_coefs, (correct,wrong)