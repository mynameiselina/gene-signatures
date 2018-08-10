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
    choose_samples,
    parse_arg_type,
    boxplot,
    set_heatmap_size,
    which_x_toPrint
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
from sklearn import svm
from distutils.util import strtobool
from scipy.stats import binom_test
from sklearn.externals import joblib

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

script_fname = os.path.basename(__file__).rsplit('.')[0]
script_path = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def _run_classification(dat, dat_target, random_state):

    logger.info("Running classification...")
    dat = dat.copy()
    dat_target = dat_target.copy()
    np.random.seed(random_state)

    # logger.info("random state = "+str(random_state))
    # model = svm.SVC(
    #     kernel='linear', random_state=random_state,
    # )
    logger.info(
        "model: svm.LinearSVC with l2 penalty and squared_hinge loss" +
        "random_state: "+str(random_state)
    )
    model = svm.LinearSVC(
        penalty='l2', C=1, random_state=random_state,
        loss='squared_hinge', dual=False
    )
    # model = linear_model.LogisticRegression(
    #     penalty='l2', C=1, random_state=random_state,
    #     solver='liblinear'
    # )

    logger.info(
        "\n\n\'we fit the model n times, where n = number of samples (" +
        str(dat.shape[0])+"), to predict each sample once\n" +
        "at the end we return the feature coefficients from each model " +
        "and a count of how many correct vs. wrong predictions we had " +
        "these counts will be evaluated with a bionomial test " +
        "and show how likely it is to get them by chance\'\n\n"
    )
    estimators = []
    correct = 0
    wrong = 0
    all_coefs = np.zeros(dat.shape)
    for choose_sample in range(dat.shape[0]):
        X = dat.drop(dat.index[choose_sample:choose_sample+1])
        y = dat_target.drop(dat.index[choose_sample:choose_sample+1])
        one_sample = dat.iloc[choose_sample:choose_sample+1, :]
        y_real = dat_target.iloc[choose_sample:choose_sample+1]

        # np.random.seed(random_state)
        model.fit(X, y)

        all_coefs[choose_sample:choose_sample+1, :] = model.coef_[0]

        y_pred = model.predict(one_sample)

        if (y_pred[0] == y_real.values[0]):
            correct = correct + 1
        else:
            wrong = wrong + 1

    # binomial test
    binom_test_thres = 0.5
    pval = binom_test(correct, n=correct+wrong, p=binom_test_thres)
    printed_results = 'correct = '+str(correct)+', wrong = '+str(wrong) + \
        '\n'+'pvalue = '+str(pval)
    logger.info(printed_results)
    logger.info("Finished classification!")

    # at the end return the model from all samples
    X = dat
    y = dat_target
    model.fit(X, y)

    return model, all_coefs, printed_results


def classification(**set_up_kwargs):
    # initialize script params
    saveReport = parse_arg_type(
        set_up_kwargs.get('saveReport', False),
        bool
    )
    toPrint = parse_arg_type(
        set_up_kwargs.get('toPrint', False),
        bool
    )
    reportName = set_up_kwargs.get('reportName', script_fname)
    txt_label = set_up_kwargs.get('txt_label', 'test_txt_label')
    sample_class_column = set_up_kwargs.get('sample_class_column', None)
    if sample_class_column is None:
        logger.error("NO class label was defined!")
        raise
    class_labels = set_up_kwargs.get('class_labels', None)
    if class_labels is not None:
        if ',' in class_labels:
            class_labels = class_labels.rsplit(',')
    class_values = set_up_kwargs.get('class_values', None)
    if class_values is not None:
        if ',' in class_values:
            class_values = class_values.rsplit(',')
            class_values = np.array(class_values).astype(int)

    random_state = set_up_kwargs.get('random_state', '0')
    if isinstance(random_state, str) and ',' in random_state:
        _random_state_or = random_state[:]
        random_state = os.path.join(*random_state.rsplit(','))
        try:
            random_state = np.array(random_state).astype(int)
        except Exception as ex:
            logger.warning(
                "invalid random state values given: "+_random_state_or +
                "\nrandom_state set to zero\n"+str(ex)
            )
    else:
        random_state = parse_arg_type(random_state, int)

    # plotting params
    plot_kwargs = set_up_kwargs.get('plot_kwargs', {})
    highRes = parse_arg_type(
        plot_kwargs.get('highRes', False),
        bool
    )
    if highRes:
        img_ext = '.pdf'
    else:
        img_ext = '.png'
    cmap_custom = plot_kwargs.get('cmap_custom', None)
    vmin = parse_arg_type(
        plot_kwargs.get('vmin', None),
        int
    )
    vmax = parse_arg_type(
        plot_kwargs.get('vmax', None),
        int
    )
    if (cmap_custom is None) and (vmin is not None) and (vmax is not None):
        custom_div_cmap_arg = abs(vmin)+abs(vmax)
        if (vmin <= 0) and (vmax >= 0):
            custom_div_cmap_arg = custom_div_cmap_arg + 1
        mincol = plot_kwargs.get('mincol', None)
        midcol = plot_kwargs.get('midcol', None)
        maxcol = plot_kwargs.get('maxcol', None)
        if (
                (mincol is not None) and
                (midcol is not None) and
                (maxcol is not None)
                ):
            cmap_custom = custom_div_cmap(
                numcolors=custom_div_cmap_arg,
                mincol=mincol, midcol=midcol, maxcol=maxcol)
        else:
            cmap_custom = custom_div_cmap(numcolors=custom_div_cmap_arg)

    # initialize directories
    MainDataDir = os.path.join(script_path, '..', 'data')

    # data input
    data_fpath = set_up_kwargs.get('data_fpath')
    if ',' in data_fpath:
        data_fpath = os.path.join(*data_fpath.rsplit(','))
        data_fpath = os.path.join(MainDataDir, data_fpath)

    # sample info input
    sample_info_fpath = set_up_kwargs.get('sample_info_fpath')
    if ',' in sample_info_fpath:
        sample_info_fpath = os.path.join(*sample_info_fpath.rsplit(','))
    sample_info_fpath = os.path.join(MainDataDir, sample_info_fpath)
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})

    # data output
    output_directory = set_up_kwargs.get('output_directory')
    if ',' in output_directory:
        output_directory = os.path.join(*output_directory.rsplit(','))
    output_directory = set_directory(
        os.path.join(MainDataDir, output_directory, reportName)
    )

    # save the set_up_kwargs in the output dir for reproducibility
    fname = 'set_up_kwargs.json'
    f = os.path.join(output_directory, fname)
    if toPrint:
        logger.info(
            '-save set_up_kwargs dictionary for reproducibility in: '+f)
    with open(f, 'w') as fp:
        json.dump(set_up_kwargs, fp, indent=4)

    # load data
    try:
        data = pd.read_csv(data_fpath, sep='\t', header=0, index_col=0)
        logger.error('loaded data file with shape: '+str(data.shape))
    except:
        logger.error('failed to read data file from: '+str(data_fpath))
        raise

    # load info table of samples
    try:
        info_table = load_clinical(
            sample_info_fpath, **sample_info_read_csv_kwargs)
    except:
        logger.error('Load info table of samples FAILED!')
        raise

    # set the ground truth
    ground_truth = info_table.loc[data.index, sample_class_column]

    # Classification
    model, all_coefs, printed_results = _run_classification(
        data, ground_truth, random_state)

    # Save to model in the output_directory
    fname = 'joblib_model.pkl'
    fpath = os.path.join(output_directory, fname)
    logger.info('-save model with joblib')
    joblib.dump(model, fpath)

    # get the genes with the nnz coefficients in classification
    clasification_results = pd.DataFrame(index=data.columns.values)
    clasification_results['classification'] = 0
    clasification_results['classification_mean_coef'] = \
        abs(all_coefs).mean(axis=0)
    clasification_results['classification_std_coef'] =\
        abs(all_coefs).std(axis=0)
    nnz_coef_bool = (abs(all_coefs).max(axis=0) > 0)
    nnz_coef_gene_names = data.columns.values[nnz_coef_bool]
    clasification_results.loc[nnz_coef_gene_names, 'classification'] = 1
    n_names = nnz_coef_gene_names.shape[0]

    logger.info('printing the names of the '+str(n_names) +
                ' features with non-zero coefficient in at least ' +
                'one classification run:\n'+str(nnz_coef_gene_names))

    # save as tab-delimited csv file
    fname = 'clasification_results.csv'
    fpath = os.path.join(output_directory, fname)
    logger.info("-save the clasification results in :\n"+fpath)
    clasification_results.to_csv(
        fpath, sep='\t', header=True, index=True)

    # save also as excel file
    fname = 'clasification_results.xlsx'
    fpath = os.path.join(output_directory, fname)
    logger.info('-save csv file as excel too')
    writer = pd.ExcelWriter(fpath)
    clasification_results.to_excel(writer)
    writer.save()

    # boxplot
    coefs_to_plot = all_coefs[:, np.where(nnz_coef_bool)[0]]
    boxplot(coefs_to_plot, coefs_to_plot.shape[1], nnz_coef_gene_names,
            title=txt_label, txtbox=printed_results, sidespace=2,
            swarm=False, n_names=n_names)
    if saveReport:
        logger.info('Save boxplot')
        fpath = os.path.join(
            output_directory, 'Fig_boxplot'+img_ext
        )
        plt.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # swarmplot
    boxplot(coefs_to_plot, coefs_to_plot.shape[1], nnz_coef_gene_names,
            title=txt_label, txtbox=printed_results, sidespace=2,
            swarm=True, n_names=n_names)
    if saveReport:
        logger.info('Save swarmplot')
        fpath = os.path.join(
            output_directory, 'Fig_swarmplot'+img_ext
        )
        plt.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # heatmap of genes with nnz coefs in classification
    try:
        yticklabels = ground_truth.index.values+',' + \
            ground_truth.values.astype(int).flatten().astype(str)
    except:
        yticklabels = ground_truth.index.values+',' + \
            ground_truth.values.flatten().astype(str)

    _, xlabels = which_x_toPrint(
        all_coefs, data.columns.values, n_names=n_names)

    fs_x, fs_y, _show_gene_names, _ = set_heatmap_size(data)
    plt.figure(figsize=(fs_x, fs_y))
    ax = sns.heatmap(data.loc[:, xlabels], vmin=vmin, vmax=vmax,
                     yticklabels=yticklabels,
                     xticklabels=_show_gene_names,
                     cmap=cmap_custom, cbar=True)
    plt.xticks(rotation=90)
    plt.title(
        str(n_names)+' selected genes from classification coefficients:' +
        class_labels[0]+'['+str(class_values[0])+'] vs. ' +
        class_labels[1]+'['+str(class_values[1])+']'
    )

    if saveReport:
        logger.info('Save heatmap')
        fpath = os.path.join(
            output_directory, 'Fig_heatmap_with_nnz_coefs'+img_ext
        )
        plt.savefig(fpath,
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()
