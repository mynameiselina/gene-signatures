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


def _run_classification(dat, dat_target):
    model = linear_model.LogisticRegression(penalty='l1', C=1, random_state=0)
    estimators = []
    correct = 0
    wrong = 0
    all_coefs = np.zeros(dat.shape)
    for choose_sample in range(dat.shape[0]):
        X = dat.drop(dat.index[choose_sample:choose_sample+1])
        y = dat_target.drop(dat.index[choose_sample:choose_sample+1])
        one_sample = dat.iloc[choose_sample:choose_sample+1, :]
        y_real = dat_target.iloc[choose_sample:choose_sample+1]

        model.fit(X, y)

        all_coefs[choose_sample:choose_sample+1, :] = model.coef_[0]

        y_pred = model.predict(one_sample)

        if (y_pred[0] == y_real.values[0]):
            correct = correct + 1
        else:
            wrong = wrong + 1

    return all_coefs, (correct, wrong)


def classification(**set_up_kwargs):
    # chose sample set from data
    # function: choose_samples()
    select_samples_from = set_up_kwargs.get('select_samples_from', None)
    select_samples_which = parse_arg_type(
        set_up_kwargs.get('select_samples_which', None),
        int
    )
    select_samples_sort_by = set_up_kwargs.get('select_samples_sort_by',
                                               'TP53_mut5,FOXA1_mut5')
    select_samples_sort_by_list = select_samples_sort_by.rsplit(',')
    select_samples_title = set_up_kwargs.get('select_samples_title',
                                             'select_all')
    clinical_label = select_samples_sort_by_list[0]
    class_labels = [clinical_label+'WT', clinical_label+'MUT']
    class_values = [0, 1]  # WT:0, MUT:1

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
    input_fname = set_up_kwargs.get('input_fname',
                                    'data_processed.csv')
    gene_info_fname = set_up_kwargs.get('gene_info_fname',
                                        'gene_info_fname.csv')
    chr_col = set_up_kwargs.get('chr_col', 'chr_int')
    gene_id_col = set_up_kwargs.get('gene_id_col', 'gene')
    sample_info_fname = set_up_kwargs.get('sample_info_fname',
                                          '20180704_emca.csv')
    sample_info_table_index_colname = \
        set_up_kwargs.get('sample_info_table_index_colname',
                          'Oncoscan_ID')
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})
    rename_genes = set_up_kwargs.get('rename_genes', 'newGeneName')

    # plotting params
    plot_kwargs = set_up_kwargs.get('plot_kwargs', {})
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
        if (vmin < 0) or (vmax < 0):
            custom_div_cmap_arg = custom_div_cmap_arg + 1
        cmap_custom = custom_div_cmap(custom_div_cmap_arg)

    highRes = parse_arg_type(
        plot_kwargs.get('highRes', False),
        bool
    )
    if highRes:
        img_ext = '.pdf'
    else:
        img_ext = '.png'

    # initialize directories
    MainDataDir = os.path.join(script_path, '..', 'data')

    # data input
    input_directory = set_up_kwargs.get('input_directory')
    if ',' in input_directory:
        input_directory = os.path.join(*input_directory.rsplit(','))
    input_directory = os.path.join(MainDataDir, input_directory)

    # sample info input
    sample_info_directory = set_up_kwargs.get('sample_info_directory')
    if ',' in sample_info_directory:
        sample_info_directory = os.path.join(
            *sample_info_directory.rsplit(','))
    sample_info_directory = os.path.join(MainDataDir, sample_info_directory)

    # gene info input
    gene_info_directory = set_up_kwargs.get('gene_info_directory')
    if ',' in gene_info_directory:
        gene_info_directory = os.path.join(
            *gene_info_directory.rsplit(','))
    gene_info_directory = os.path.join(MainDataDir, gene_info_directory)

    # selected genes input
    selected_genes_directory = set_up_kwargs.get('selected_genes_directory')
    if ',' in selected_genes_directory:
        selected_genes_directory = os.path.join(
            *selected_genes_directory.rsplit(','))
    selected_genes_directory = os.path.join(
        MainDataDir, selected_genes_directory)

    # data output
    output_directory = set_up_kwargs.get('output_directory')
    if output_directory is None:
        output_directory = set_directory(
            os.path.join(input_directory, reportName)
        )
    else:
        if ',' in output_directory:
            output_directory = os.path.join(*output_directory.rsplit(','))
        output_directory = set_directory(
            os.path.join(MainDataDir, output_directory, reportName)
        )

    # load info table of samples
    if toPrint:
        logger.info('Load info table of samples')
    fpath = os.path.join(sample_info_directory, sample_info_fname)
    info_table = load_clinical(fpath,
                               col_as_index=sample_info_table_index_colname,
                               **sample_info_read_csv_kwargs)

    # load data
    fpath = os.path.join(input_directory, input_fname)
    data = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
    empty_pat = data.sum(axis=1).isnull()
    if empty_pat.any():
        logger.info('Patients with missing values in all genes: ' +
                    str(data.index[empty_pat]))

    # keep only info_table with data
    ids_tmp = set(info_table.index.values
                  ).intersection(set(data.index.values))
    info_table = info_table.loc[ids_tmp].copy()
    # info_table = info_table.reset_index()

    # load gene info
    if gene_info_fname is not None:
        fpath = os.path.join(gene_info_directory, gene_info_fname)
        genes_positions_table = pd.read_csv(fpath, sep='\t', header=0,
                                            index_col=0)
        # get gene chrom position
        xlabels, xpos = get_chr_ticks(genes_positions_table, data,
                                      id_col='gene', chr_col=chr_col)
    else:
        xlabels, xpos = None, None

    # select the samples for the chosen comparison (e.g. all, only TP53wt, etc)
    logger.info('select_samples_from: '+str(select_samples_from) +
                ', select_samples_which: '+str(select_samples_which) +
                ', select_samples_sort_by: '+str(select_samples_sort_by) +
                ', select_samples_title: '+str(select_samples_title))

    ids_tmp = choose_samples(info_table.reset_index(),
                             sample_info_table_index_colname,
                             choose_from=select_samples_from,
                             choose_what=select_samples_which,
                             sortby=select_samples_sort_by_list,
                             ascending=False)

    # keep a subpart of the info_table (rows and columns)
    info_table = info_table.loc[ids_tmp][
            select_samples_sort_by_list].copy()
    info_table = info_table.dropna()
    # create the row labels for the plots
    pat_labels_txt = info_table.astype(int).reset_index().values

    # keep only these samples from the data
    data = data.loc[info_table.index, :].copy()

    # load selected genes
    fname = 'diff_genes_selected_'+select_samples_title+'.csv'
    fpath = os.path.join(selected_genes_directory, fname)
    diff_genes = pd.read_csv(fpath, sep='\t', header=0, index_col=0)

    title = select_samples_title
    select_genes = diff_genes.index.values
    data = data.reindex(select_genes, axis=1)

    if rename_genes is not None:
        gene_newNames = diff_genes[rename_genes]

        # get the selected genes from the data
        data = data.T
        data[rename_genes] = gene_newNames
        data.reset_index(drop=True, inplace=True)
        data.set_index(rename_genes, inplace=True)
        data = data.T

    else:
        gene_newNames = None

    ground_truth = info_table.reindex(data.index)[clinical_label]

    # Classification
    binom_test_thres = 0.5
    logger.info("Running classification...")
    all_coefs, (correct, wrong) = _run_classification(data, ground_truth)
    pval = binom_test(correct, n=correct+wrong, p=binom_test_thres)
    printed_results = 'correct = '+str(correct)+', wrong = '+str(wrong) + \
        '\n'+'pvalue = '+str(pval)
    logger.info(printed_results)
    logger.info("Finished classification!")

    sum_thres = abs(all_coefs).mean()*all_coefs.shape[0]
    n_names = (abs(all_coefs).sum(axis=0) > sum_thres).sum()
    logger.info('figures: printing the names from the first ' +
                str(n_names)+' features with the highest absolute ' +
                'coefficients across all classification runs')
    # boxplot
    boxplot(all_coefs, data.shape[1], data.columns.values,
            title=title, txtbox=printed_results, sidespace=3,
            swarm=False, n_names=n_names)
    if saveReport:
        logger.info('Save boxplot')
        fpath = os.path.join(
            output_directory, 'Fig_'+title+'_boxplot'+img_ext
        )
        plt.savefig(fpath, transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # swarmplot
    boxplot(all_coefs, data.shape[1], data.columns.values,
            title=title, txtbox=printed_results, sidespace=2,
            swarm=True, n_names=n_names)
    if saveReport:
        logger.info('Save swarmplot')
        fpath = os.path.join(
            output_directory, 'Fig_'+title+'_swarmplot'+img_ext
        )
        plt.savefig(fpath, transparent=True,
                    bbox_inches='tight', pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # heatmap all selected genes
    xpos, xlabels = which_x_toPrint(
        all_coefs, data.columns.values, n_names=n_names)
    xpos = xpos + 0.5
    if data.shape[1] < 6:
        figsize = (8, 8)
    elif data.shape[1] < 15:
        figsize = (15, 8)
    else:
        figsize = (25, 8)

    plt.figure(figsize=figsize)
    ticklabels = ground_truth.index.values+',' + \
        ground_truth.values.flatten().astype(str)
    bwr_custom = custom_div_cmap(5)
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax,
                     yticklabels=ticklabels, xticklabels=False,
                     cmap=cmap_custom, cbar=True)
    plt.xticks(xpos, xlabels, rotation=90)
    plt.title(title+' diff mutated genes')

    if saveReport:
        logger.info('Save heatmap')
        fpath = os.path.join(
            output_directory, 'Fig_'+title+'_heatmap'+img_ext
        )
        plt.savefig(fpath,
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # heatmap some selected genes (according to the sum_thres)
    _, xlabels = which_x_toPrint(
        all_coefs, data.columns.values, n_names=n_names)
    if data.shape[1] < 6:
        figsize = (8, 8)
    elif data.shape[1] < 15:
        figsize = (15, 8)
    else:
        figsize = (25, 8)

    plt.figure(figsize=figsize)
    ticklabels = ground_truth.index.values+',' + \
        ground_truth.values.flatten().astype(str)
    ax = sns.heatmap(data.loc[:, xlabels], vmin=vmin, vmax=vmax,
                     yticklabels=ticklabels, xticklabels=True,
                     cmap=cmap_custom, cbar=True)
    plt.title(title+' diff mutated genes - first '+str(n_names) +
              ' selected from classification coefficients')

    if saveReport:
        logger.info('Save heatmap')
        fpath = os.path.join(
            output_directory, 'Fig_'+title+'_heatmap2'+img_ext
        )
        plt.savefig(fpath, transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()
