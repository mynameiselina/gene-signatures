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
    dat = dat.copy()
    dat_target = dat_target.copy()
    np.random.seed(random_state)
    # logger.info("random state = "+str(random_state))
    # model = svm.SVC(
    #     kernel='linear', random_state=random_state,
    # )
    model = svm.LinearSVC(
        penalty='l2', C=1, random_state=random_state,
        loss='squared_hinge', dual=False
    )
    # model = linear_model.LogisticRegression(
    #     penalty='l2', C=1, random_state=random_state,
    #     solver='liblinear'
    # )
    estimators = []
    correct = 0
    wrong = 0
    all_coefs = np.zeros(dat.shape)
    for choose_sample in range(dat.shape[0]):
        X = dat.drop(dat.index[choose_sample:choose_sample+1])
        y = dat_target.drop(dat.index[choose_sample:choose_sample+1])
        one_sample = dat.iloc[choose_sample:choose_sample+1, :]
        y_real = dat_target.iloc[choose_sample:choose_sample+1]

        np.random.seed(random_state)
        model.fit(X, y)

        all_coefs[choose_sample:choose_sample+1, :] = model.coef_[0]

        y_pred = model.predict(one_sample)

        if (y_pred[0] == y_real.values[0]):
            correct = correct + 1
        else:
            wrong = wrong + 1

    return all_coefs, (correct, wrong)


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
    sample_class_label = set_up_kwargs.get('sample_class_label', None)
    if sample_class_label is NOne:
        logger.error("NO class label was defined!")
        raise
    random_state = parse_arg_type(
        set_up_kwargs.get('random_state', 0),
        int
    )

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
    # cmap_custom = plot_kwargs.get('cmap_custom', None)
    # vmin = parse_arg_type(
    #     plot_kwargs.get('vmin', None),
    #     int
    # )
    # vmax = parse_arg_type(
    #     plot_kwargs.get('vmax', None),
    #     int
    # )
    # if (cmap_custom is None) and (vmin is not None) and (vmax is not None):
    #     custom_div_cmap_arg = abs(vmin)+abs(vmax)
    #     if (vmin <= 0) and (vmax >= 0):
    #         custom_div_cmap_arg = custom_div_cmap_arg + 1
    #     mincol = plot_kwargs.get('mincol', None)
    #     midcol = plot_kwargs.get('midcol', None)
    #     maxcol = plot_kwargs.get('maxcol', None)
    #     if (
    #             (mincol is not None) and
    #             (midcol is not None) and
    #             (maxcol is not None)
    #             ):
    #         cmap_custom = custom_div_cmap(
    #             numcolors=custom_div_cmap_arg,
    #             mincol=mincol, midcol=midcol, maxcol=maxcol)
    #     else:
    #         cmap_custom = custom_div_cmap(numcolors=custom_div_cmap_arg)

    # initialize directories
    MainDataDir = os.path.join(script_path, '..', 'data')

    # data input
    data_fpath = set_up_kwargs.get('data_fpath')
    if ',' in sample_info_fpath:
        data_fpath = os.path.join(*data_fpath.rsplit(','))

    # sample info input
    sample_info_fpath = set_up_kwargs.get('sample_info_fpath')
    if ',' in sample_info_fpath:
        sample_info_fpath = os.path.join(*sample_info_fpath.rsplit(','))
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})

    # data output
    output_directory = set_up_kwargs.get('output_directory')
    if ',' in output_directory:
        output_directory = os.path.join(*output_directory.rsplit(','))
    output_directory = set_directory(
        os.path.join(MainDataDir, output_directory, reportName)
    )

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
        logger.error('loaded data file with shape: '+str(df.shape))
    except:
        logger.error('failed to read data file from: '+str(fpath))
        raise

    # load info table of samples
    try:
        info_table = load_clinical(
            sample_info_fpath, **sample_info_read_csv_kwargs)
    except:
        logger.error('Load info table of samples FAILED!')
        raise

    # set the ground truth
    ground_truth = info_table.loc[data.index, sample_class_label]

    # Classification
    binom_test_thres = 0.5
    logger.info("Running classification...")
    all_coefs, (correct, wrong) =\
        _run_classification(data, ground_truth, random_state)
    pval = binom_test(correct, n=correct+wrong, p=binom_test_thres)
    printed_results = 'correct = '+str(correct)+', wrong = '+str(wrong) + \
        '\n'+'pvalue = '+str(pval)
    logger.info(printed_results)
    logger.info("Finished classification!")

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
    yticklabels = ground_truth.index.values+',' + \
        ground_truth.values.flatten().astype(str)

    _, xlabels = which_x_toPrint(
        all_coefs, data.columns.values, n_names=n_names)

    ds_y, ds_x = data.shape
    fs_x = 25 if ds_x > 45 else 15 if ds_x > 30 else 10 if ds_x > 3 else 5
    fs_y = 20 if ds_y > 40 else 15 if ds_y > 30 else 10

    plt.figure(figsize=(fs_x, fs_y))
    ax = sns.heatmap(data.loc[:, xlabels], vmin=vmin, vmax=vmax,
                     yticklabels=yticklabels,
                     xticklabels=True,
                     cmap=cmap_custom, cbar=True)
    plt.xticks(rotation=90)
    plt.title(str(n_names)+' selected genes from classification coefficients')

    if saveReport:
        logger.info('Save heatmap')
        fpath = os.path.join(
            output_directory, 'Fig_heatmap_with_'+sample_final_id+'_id'+img_ext
        )
        plt.savefig(fpath,
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()
