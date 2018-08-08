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
    get_chr_ticks,
    distplot_breakYaxis,
    parse_arg_type,
    set_heatmap_size
)

# basic imports
import os
import numpy as np
import pandas as pd
import scipy as sp
import math
import json
import logging
from scipy.spatial.distance import (
    pdist, squareform
)
from distutils.util import strtobool

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
    select_samples_from = set_up_kwargs.get('select_samples_from', None)
    select_samples_which = parse_arg_type(
        set_up_kwargs.get('select_samples_which', None),
        int
    )
    select_samples_sort_by = set_up_kwargs.get('select_samples_sort_by',
                                               None)
    if select_samples_sort_by is not None:
        select_samples_sort_by = select_samples_sort_by.rsplit(',')
    select_samples_title = set_up_kwargs.get('select_samples_title',
                                             'select_all')

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
                                        None)
    chr_col = set_up_kwargs.get('chr_col', 'chr_int')
    gene_id_col = set_up_kwargs.get('gene_id_col', 'gene')
    sample_info_fname = set_up_kwargs.get('sample_info_fname',
                                          None)
    if ',' in sample_info_fname:
        sample_info_fname = os.path.join(*sample_info_fname.rsplit(','))
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})

    # plotting params
    plot_kwargs = set_up_kwargs.get('plot_kwargs', {})
    function_dict = plot_kwargs.get('function_dict', None)
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
    if gene_info_directory is None:
        gene_info_directory = input_directory
    else:
        if ',' in gene_info_directory:
            gene_info_directory = os.path.join(
                *gene_info_directory.rsplit(','))
            gene_info_directory = os.path.join(
                MainDataDir, gene_info_directory)

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
    # save the set_up_kwargs in the output dir for reproducibility
    fname = 'set_up_kwargs.json'
    f = os.path.join(output_directory, fname)
    if toPrint:
        logger.info(
            '-save set_up_kwargs dictionary for reproducibility in: '+f)
    with open(f, 'w') as fp:
        json.dump(set_up_kwargs, fp, indent=4)

    # pairwise distances params
    compute_pdist = parse_arg_type(
        set_up_kwargs.get('compute_pdist', False),
        bool
    )
    pdist_fname = 'data_'+select_samples_title+'__genes_pdist.h5'
    pdist_fpath = os.path.join(input_directory, pdist_fname)
    if not os.path.exists(pdist_fpath):
        compute_pdist = True

    # load info table of samples
    if toPrint:
        logger.info('Load info table of samples')
    fpath = os.path.join(sample_info_directory, sample_info_fname)
    info_table = load_clinical(fpath,  **sample_info_read_csv_kwargs)

    # load input_data
    fpath = os.path.join(input_directory, input_fname)
    input_data = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
    empty_pat = input_data.sum(axis=1).isnull()
    if empty_pat.any():
        logger.info('Patients with missing values in all genes: ' +
                    str(input_data.index[empty_pat]))
    input_data = input_data.fillna(0)

    # keep only info_table with input_data
    ids_tmp = set(info_table.index.values
                  ).intersection(set(input_data.index.values))
    info_table = info_table.loc[ids_tmp].copy()
    # info_table = info_table.reset_index()

    # load gene info
    fpath = os.path.join(gene_info_directory, gene_info_fname)
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
    ids_tmp = choose_samples(info_table.reset_index(),
                             info_table.index.name,
                             choose_from=select_samples_from,
                             choose_what=select_samples_which,
                             sortby=select_samples_sort_by,
                             ascending=False)

    pat_labels = info_table.loc[ids_tmp][
            select_samples_sort_by].copy()
    pat_labels = pat_labels.dropna()
    pat_labels_txt = pat_labels.astype(int).reset_index().values
    pat_labels_title = str(pat_labels.reset_index().columns.values)

    data = input_data.loc[pat_labels.index, :].copy()

    # REMOVE DUPLICATES!!!!
    uniqdata, dupldict, _, _ = remove_andSave_duplicates(
        data, to_compute_euclidean_distances=compute_pdist,
        to_save_euclidean_distances=saveReport, to_save_output=saveReport,
        output_filename=input_fname.rsplit('.')[0]+'__'+select_samples_title,
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
            logger.info('Plotting distplot..')
            sns.distplot(choose_data.values.flatten(),
                         hist=True, kde=False, color='b')
            plt.title("Copy number abundance in "+txt_label+" (uniq genes)")
            if saveReport:
                logger.info('Save distplot')
                plt.savefig(os.path.join(
                    output_directory, 'Fig_distplot_' +
                    select_samples_title+fext[i_data]+img_ext),
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
                plt.close("all")
            else:
                plt.show()

            # distplot break Y-axis
            logger.info('Plotting break Y-axis distplot..')
            _, uniq_count = np.unique(choose_data.values.flatten(),
                                      return_counts=True)
            ymax_bottom = int(math.ceil(
                np.sort(uniq_count)[-2] / 1000.0)
                ) * 1000
            ymax_top = int(math.ceil(
                np.sort(uniq_count)[-1] / 10000.0)
                ) * 10000
            distplot_breakYaxis(choose_data.values, ymax_bottom,
                                ymax_top, color='b', d=0.005,
                                pad=1.5, figsize=(10, 6),
                                mytitle='Copy number abundance in '+txt_label +
                                        'with cropped y axis (uniq genes)')
            if saveReport:
                logger.info('Save distplot')
                plt.savefig(os.path.join(
                    output_directory, 'Fig_distplot_breakYaxis_' +
                    select_samples_title+fext[i_data]+img_ext),
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
                plt.close("all")
            else:
                plt.show()

        # Plot heatmap
        _figure_x_size, _figure_y_size, _, _ = \
            set_heatmap_size(data)
        plt.figure(figsize=(_figure_x_size, _figure_y_size))
        ax = sns.heatmap(choose_data, vmin=vmin, vmax=vmax,
                         yticklabels=pat_labels_txt, xticklabels=False,
                         cmap=cmap_custom, cbar=False)
        plt.xticks(xpos_choose[i_data], xlabels_choose[i_data], rotation=0)
        plt.xlabel('chromosomes (the number is aligned at the end ' +
                   'of the chr region)')
        plt.ylabel('samples '+select_samples_title+'\n'+pat_labels_title)
        plt.xticks(rotation=90)
        cbar = ax.figure.colorbar(ax.collections[0])
        if function_dict is not None:
            functionImpact_dict_r = dict(
                (v, k) for k, v in function_dict.items()
                )
            myTicks = [0, 1, 2, 3, 4, 5]
            cbar.set_ticks(myTicks)
            cbar.set_ticklabels(pd.Series(myTicks).map(functionImpact_dict_r))
        else:
            if custom_div_cmap_arg is not None:
                cbar.set_ticks(np.arange(-custom_div_cmap_arg,
                                         custom_div_cmap_arg))

        plt.title(
            txt_label+'\nheatmap of ' +
            select_samples_title+' samples')

        if saveReport:
            logger.info('Save heatmap')
            plt.savefig(os.path.join(
                output_directory, 'Fig_heatmap_'+select_samples_title +
                fext[i_data]+img_ext),
                transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

        # Plot pairwise sample correlations
        data_cor = 1-squareform(pdist(choose_data, 'correlation'))
        plt.figure(figsize=(15, 10))
        sns.heatmap(data_cor, vmin=-1, vmax=1, yticklabels=pat_labels_txt,
                    xticklabels=pat_labels_txt, cmap='PiYG', square=True)
        plt.xlabel("samples "+select_samples_title)
        plt.ylabel(pat_labels_title)
        plt.title("Auto-corerelation of "+select_samples_title +
                  " samples - "+txt_label)
        if saveReport:
            logger.info('Save heatmap')
            plt.savefig(os.path.join(
                output_directory, 'Fig_corr_'+select_samples_title +
                fext[i_data]+img_ext),
                transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()
