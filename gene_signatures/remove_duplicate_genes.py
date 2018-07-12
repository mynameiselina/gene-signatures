# custom imports
from omics_processing.io import (
    set_directory, load_clinical
)
from omics_processing.remove_duplicates import (
    remove_andSave_duplicates
)
from gene_signatures.core import (
    choose_samples,
    custom_div_cmap,
    get_chr_ticks
)

# basic imports
import os
import numpy as np
import pandas as pd
import scipy as sp
import math
import logging

# plotting imports
import matplotlib
import seaborn as sns
import matplotlib.pyplot as plt
sns.set_style('white')
sns.set_context('poster')
default_palette = sns.color_palette()
default_backend = plt.get_backend()

script_path = os.path.dirname(__file__)
script_fname = os.path.basename(__file__).rsplit('.')[0]
logger = logging.getLogger(__name__)


def remove_duplicate_genes(**set_up_kwargs):

    # chose sample set from data
    # function: choose_samples()
    select_samples_from = set_up_kwargs.get('select_samples_from', None)
    select_samples_which = set_up_kwargs.get('select_samples_which', None)
    select_samples_sort_by = set_up_kwargs.get('select_samples_sort_by',
                                               'TP53_mut5,FOXA1_mut5')
    select_samples_sort_by_list = select_samples_sort_by.rsplit(',')
    select_samples_title = set_up_kwargs.get('select_samples_title',
                                             'select_all')

    # initialize script params
    saveReport = set_up_kwargs.get('saveReport', False)
    toPrint = set_up_kwargs.get('toPrint', False)
    reportName = set_up_kwargs.get('reportName', script_fname)
    txt_label = set_up_kwargs.get('txt_label', 'test_txt_label')
    input_fname = set_up_kwargs.get('input_fname',
                                    'table_ordered.csv')
    gene_info_fname = set_up_kwargs.get('gene_info_fname',
                                        'gene_info_fname.csv')
    chr_col = set_up_kwargs.get('chr_col', 'chr_int')
    gene_id_col = set_up_kwargs.get('gene_id_col', 'gene')
    sample_info_fname = set_up_kwargs.get('sample_info_fname',
                                          '20180704_emca.csv')
    sample_info_table_index_colname = \
        set_up_kwargs.get('sample_info_table_index_colname',
                          'Oncoscan_ID')

    # plotting params
    plot_kwargs = set_up_kwargs.get('plot_kwargs', {})
    cmap_custom = plot_kwargs.get('cmap_custom', None)
    vmin = plot_kwargs.get('vmin', None)
    vmax = plot_kwargs.get('vmax', None)

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
        sample_info_directory = os.path.join(*sample_info_directory.rsplit(','))
    sample_info_directory = os.path.join(MainDataDir, sample_info_directory)
    # data output
    output_directory = set_up_kwargs.get('output_directory')
    if ',' in output_directory:
        output_directory = os.path.join(*output_directory.rsplit(','))
    output_directory = set_directory(
        os.path.join(MainDataDir, output_directory, reportName)
    )

    # pairwise distances params
    compute_pdist = set_up_kwargs.get('compute_pdist', False)
    pdist_fname = 'data_'+select_samples_title+'__genes_pdist.h5'
    pdist_fpath = os.path.join(input_directory, pdist_fname)
    if not os.path.exists(pdist_fpath):
        compute_pdist = True

    # load info table of samples
    if toPrint:
        logger.info('Load info table of samples')
    fpath = os.path.join(sample_info_directory, sample_info_fname)
    info_table = load_clinical(fpath,
                               col_as_index=sample_info_table_index_colname,
                               **{'na_values': ' '})

    # load input_fname
    fpath = os.path.join(input_directory, input_fname)
    input_data = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
    empty_pat = input_data.sum(axis=1).isnull()
    if empty_pat.any():
        logger.info('Patients with missing values in all genes: ' +
                    str(input_data.index[empty_pat]))

    # keep only info_table with input_data
    ids_tmp = set(info_table.index.values
                  ).intersection(set(input_data.index.values))
    info_table = info_table.loc[ids_tmp].copy()
    # info_table = info_table.reset_index()

    # load gene info
    fpath = os.path.join(input_directory, gene_info_fname)
    genes_positions_table = pd.read_csv(fpath, sep='\t', header=0,
                                        index_col=0)
    # get gene chrom position
    xlabels, xpos = get_chr_ticks(genes_positions_table, input_data,
                                  id_col='gene', chr_col=chr_col)

    logger.info('select_samples_from: '+str(select_samples_from) +
                'select_samples_which: '+str(select_samples_which) +
                'select_samples_sort_by: '+str(select_samples_sort_by) +
                'select_samples_title: '+str(select_samples_title))

    # choose samples to plot heatmap and pairwise correlation
    ids_tmp = choose_samples(info_table.reset_index(), sample_info_table_index_colname,
                             choose_from=select_samples_from, choose_what=select_samples_which,
                             sortby=select_samples_sort_by_list,
                             ascending=False)

    pat_labels = info_table.loc[ids_tmp][
            select_samples_sort_by_list].copy()
    pat_labels = pat_labels.dropna()
    pat_labels_txt = pat_labels.astype(int).reset_index().values

    data = input_data.loc[pat_labels.index, :].copy()

    # REMOVE DUPLICATES!!!!
    uniqdata, dupldict, _, _ = remove_andSave_duplicates(
        data.fillna(0), to_compute_euclidean_distances=compute_pdist,
        to_save_euclidean_distances=saveReport, to_save_output=saveReport,
        output_filename=input_fname.rsplit('.')[0],
        output_directory=output_directory)
    # get gene chrom position
    xlabels_uniq, xpos_uniq = get_chr_ticks(genes_positions_table, uniqdata,
                                            id_col='gene', chr_col=chr_col)

    fext = ['', '_uniq']
    xlabels_choose = [xlabels, xlabels_uniq]
    xpos_choose = [xpos, xpos_uniq]
    for i_data, choose_data in enumerate([data, uniqdata]):
        if select_samples_which is None:
            # distplot DO NOT break Y-axis
            sns.distplot(choose_data.fillna(0).values.flatten(),
                         hist=True, kde=False, color='b')
            plt.title("Copy number abundance in "+txt_label+" (uniq genes)")
            if saveReport:
                logger.info('Save distplot')
                plt.savefig(os.join.path(
                    output_directory, 'Fig_distplot_' +
                    select_samples_title+fext[i_data]+'.png'),
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
            if not saveReport:
                plt.show()
            else:
                plt.close("all")

            # distplot break Y-axis
            _, uniq_count = np.unique(choose_data.fillna(0).values.flatten(),
                                      return_counts=True)
            ymax_bottom = int(math.ceil(
                np.sort(uniq_count)[-2] / 1000.0)
                ) * 1000
            ymax_top = int(math.ceil(
                np.sort(uniq_count)[-1] / 10000.0)
                ) * 10000
            distplot_breakYaxis(choose_data.fillna(0).values, ymax_bottom,
                                ymax_top, color='b', d=0.005,
                                pad=1.5, figsize=(10, 6),
                                mytitle='Copy number abundance in '+txt_label +
                                        'with cropped y axis (uniq genes)')
            if saveReport:
                logger.info('Save distplot')
                plt.savefig(os.path.join(
                    output_directory, 'Fig_distplot_breakYaxis_' +
                    select_samples_title+fext[i_data]+'.png'),
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
            if not saveReport:
                plt.show()
            else:
                plt.close("all")

        # Plot heatmap
        plt.figure(figsize=(20, 8))
        sns.heatmap(choose_data, vmin=vmin, vmax=vmax,
                    yticklabels=pat_labels_txt, xticklabels=False,
                    cmap=cmap_custom, cbar_kws={'ticks': np.arange(-5, 5)})
        plt.xticks(xpos_choose[i_data], xlabels_choose[i_data], rotation=0)
        plt.xlabel('chromosomes (the number is aligned at the end ' +
                   'of the chr region)')
        plt.ylabel('samples '+select_samples_title)
        plt.title(txt_label+' heatmap of '+select_samples_title +
                  ' samples (blue:amplification, red:deletion)')
        if saveReport:
            logger.info('Save heatmap')
            plt.savefig(os.path.join(
                output_directory, 'Fig_heatmap_'+select_samples_title +
                fext[i_data]+'.png'), 
                transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
        if not saveReport:
            plt.show()
        else:
            plt.close("all")

        # Plot pairwise sample correlations
        data_cor = 1-squareform(pdist(choose_data.fillna(0), 'correlation'))
        plt.figure(figsize=(15, 10))
        sns.heatmap(data_cor, vmin=-1, vmax=1, yticklabels=pat_labels_txt,
                    xticklabels=pat_labels_txt, cmap='PiYG', square=True)
        plt.xlabel("samples "+select_samples_title)
        plt.ylabel("samples "+select_samples_title)
        plt.title("Auto-corerelation of "+select_samples_title +
                  " samples - "+txt_label)
        if saveReport:
            logger.info('Save heatmap')
            plt.savefig(os.path.join(
                output_directory, 'Fig_corr_'+select_samples_title +
                fext[i_data]+'.png'), 
                transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
        if not saveReport:
            plt.show()
        else:
            plt.close("all")
