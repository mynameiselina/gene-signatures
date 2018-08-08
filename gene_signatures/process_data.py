# custom imports
from omics_processing.io import (
    set_directory, load_clinical
)
from gene_signatures.core import (
    custom_div_cmap,
    get_chr_ticks,
    parse_arg_type,
    choose_samples,
)

# basic imports
import os
import sys
import numpy as np
import pandas as pd
import json
import scipy as sp
from natsort import natsorted, index_natsorted
import logging
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


def process_data(**set_up_kwargs):
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

    input_fname = set_up_kwargs.get(
        'input_fname', 'data_processed.csv')
    gene_info_fname = set_up_kwargs.get(
        'gene_info_fname', 'gene_info_fname.csv')
    txt_label = set_up_kwargs.get('txt_label', 'test_txt_label')
    chr_col = set_up_kwargs.get('chr_col', 'chr_int')
    gene_id_col = set_up_kwargs.get('gene_id_col', 'gene')
    remove_patients = set_up_kwargs.get('remove_patients', None)
    if remove_patients is None or remove_patients == "":
        remove_patients_list = []
    else:
        remove_patients_list = remove_patients.rsplit(',')

    select_genes = set_up_kwargs.get('select_genes', None)
    if select_genes is None or select_genes == "":
        select_genes_list = []
    else:
        select_genes_list = select_genes.rsplit(',')

    sample_info_fname = set_up_kwargs.get(
        'sample_info_fname', '20180704_emca.csv')
    if ',' in sample_info_fname:
        sample_info_fname = os.path.join(*sample_info_fname.rsplit(','))
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})
    sample_info_table_index_colname = \
        set_up_kwargs.get('sample_info_table_index_colname',
                          'Oncoscan_ID')
    sample_info_table_sortLabels = \
        set_up_kwargs.get('sample_info_table_sortLabels',
                          'TP53_mut5,FOXA1_mut5')
    sample_info_table_sortLabels_list = \
        sample_info_table_sortLabels.rsplit(',')

    # chose sample set from data
    # function: choose_samples()
    select_samples_from = set_up_kwargs.get('select_samples_from', None)
    select_samples_which = parse_arg_type(
        set_up_kwargs.get('select_samples_which', None),
        int
    )
    # map_values_dict
    map_values_dict = set_up_kwargs.get('map_values_dict', {})
    if map_values_dict:
        map_values_dict = {int(k): int(v) for k, v in map_values_dict.items()}

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
            os.path.join(input_directory, reportName))
    else:
        if ',' in output_directory:
            output_directory = os.path.join(*output_directory.rsplit(','))
        output_directory = set_directory(
            os.path.join(MainDataDir, output_directory, reportName))

    # save the set_up_kwargs in the output dir for reproducibility
    fname = 'set_up_kwargs.json'
    f = os.path.join(output_directory, fname)
    if toPrint:
        logger.info(
            '-save set_up_kwargs dictionary for reproducibility in: '+f)
    with open(f, 'w') as fp:
        json.dump(set_up_kwargs, fp, indent=4)
    #########################################
    # load input_data
    fpath = os.path.join(input_directory, input_fname)
    data = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
    data = data.fillna(0)

    # load info table of samples
    if toPrint:
        logger.info('Load info table of samples')
    fpath = os.path.join(sample_info_directory, sample_info_fname)
    info_table = load_clinical(fpath,
                               col_as_index=sample_info_table_index_colname,
                               **sample_info_read_csv_kwargs)

    # keep only info_table with data
    ids_tmp = set(info_table.index.values
                  ).intersection(set(data.index.values))
    info_table = info_table.loc[ids_tmp].copy()

    # load gene info
    fpath = os.path.join(gene_info_directory, gene_info_fname)
    try:
        genes_positions_table = pd.read_csv(fpath, sep='\t', header=0,
                                            index_col=0)
        # get gene chrom position
        xlabels, xpos = get_chr_ticks(
            genes_positions_table, data,
            id_col='gene', chr_col=chr_col)
    except:
        logger.warning('could not get genes position info')
        xlabels, xpos = None, None

    #########################################
    # CHECK if there are empty genes and remove them
    is_empty = (data.isnull()).all(axis=0)
    if is_empty.any():
        genes2remove = data.columns[is_empty]
        data.drop(genes2remove, axis=1, inplace=True)
        if toPrint:
            logger.info('remove the following genes because ' +
                        'they have no values in the table: ' +
                        str(genes2remove))

    # CHECK if there are empty patients BUT don't remove them
    empty_pat = data.sum(axis=1).isnull()
    if empty_pat.any():
        logger.info('Patients with missing values in all genes: ' +
                    str(data.index[empty_pat]))

    # SELECT specific genes (optional)
    if len(select_genes_list) > 0:
        # first take intersection of with data
        select_genes_list = set(
            data.columns.values).intersection(set(select_genes_list))
        # then keep only these genes from in the data
        data = data.loc[:, select_genes_list].copy()
        if genes_positions_table is not None:
            xlabels, xpos = get_chr_ticks(
                genes_positions_table, data,
                id_col='gene', chr_col=chr_col)

    # MAP values with a dictionary (optional)
    if map_values_dict:
        _diff_set = set(np.unique(data.values.flatten().astype(int)))\
            .difference(set([0]))\
            .difference(set(list(map_values_dict.keys())))
        if _diff_set:
            logger.warning(
                "the user\'s dict to replace data values is incomplete " +
                "the following values in the data are not accounted for " +
                "and will remain the same:\n"+str(_diff_set)
            )
        logger.info(
            "replacing data values with user\'s dictionary:\n" +
            str(map_values_dict)
        )
        data.replace(map_values_dict, inplace=True)

    # SELECT sample groups (optional)
    ids_tmp = choose_samples(info_table.reset_index(),
                             sample_info_table_index_colname,
                             choose_from=select_samples_from,
                             choose_what=select_samples_which,
                             sortby=sample_info_table_sortLabels_list,
                             ascending=False)
    choose_columns = sample_info_table_sortLabels_list
    if sample_info_table_index_colname in choose_columns:
        choose_columns = list(
            set(sample_info_table_sortLabels_list)
            .difference(set([sample_info_table_index_colname])))
    pat_labels = info_table.loc[ids_tmp][choose_columns].copy()
    try:
        pat_labels_txt = pat_labels.astype(int).reset_index().values
    except:
        pat_labels_txt = pat_labels.reset_index().values
    pat_labels_title = str(pat_labels.reset_index().columns.values)

    # PLOT heatmap without gene ordering
    _show_gene_names = False
    _figure_x_size = 20
    if data.shape[1] < 50:
        _show_gene_names = True
        _figure_x_size = 10
    if toPrint:
        logger.info('Plot heatmap before gene ordering')
    plt.figure(figsize=(_figure_x_size, 8))
    ax = sns.heatmap(data.loc[pat_labels.index],
                     vmin=vmin, vmax=vmax, xticklabels=_show_gene_names,
                     yticklabels=pat_labels_txt, cmap=cmap_custom, cbar=False)
    ax.set_ylabel(pat_labels_title)
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
            cbar.set_ticks(np.arange(
                -custom_div_cmap_arg, custom_div_cmap_arg))
    if saveReport:
        if toPrint:
            logger.info('Save heatmap')
        plt.savefig(os.path.join(output_directory, 'Fig_heatmap'+img_ext),
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    #########################################
    if (xlabels is not None) and (xpos is not None):
        # ORDER genes
        if toPrint:
            logger.info('Order data according to genomic position')

        # extract the gene relative order
        gene_order = genes_positions_table.set_index(
            gene_id_col).loc[:, 'order'].copy()
        # keep only gene_order with data
        ids_tmp = set(
            gene_order.index.values).intersection(set(data.columns.values))
        # keep only the order of these genes
        gene_order = gene_order.loc[ids_tmp].copy()
        gene_order = gene_order.sort_values()
        # then keep only these genes from the data
        data = data.loc[:, gene_order.index].copy()

        # data = pd.DataFrame(data, columns=sorted(
        #     gene_order_dict, key=gene_order_dict.get))

        # PLOT heatmap after gene ordering
        _show_gene_names = False
        _figure_x_size = 20
        if data.shape[1] < 50:
            _show_gene_names = True
            _figure_x_size = 10

        if toPrint:
            logger.info('Plot heatmap after gene ordering')

        plt.figure(figsize=(_figure_x_size, 8))
        ax = sns.heatmap(
            data.loc[pat_labels.index],
            vmin=vmin, vmax=vmax, xticklabels=_show_gene_names,
            yticklabels=pat_labels_txt, cmap=cmap_custom, cbar=False)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, rotation=0)
        ax.set_ylabel(pat_labels_title)
        cbar = ax.figure.colorbar(ax.collections[0])
        myTicks = np.arange(vmin, vmax+2, 1)
        cbar.set_ticks(myTicks)
        if function_dict is not None:
            functionImpact_dict_r = dict(
                (v, k) for k, v in function_dict.items()
                )
            cbar.set_ticklabels(pd.Series(myTicks).map(functionImpact_dict_r))
        else:
            if custom_div_cmap_arg is not None:
                cbar.set_ticks(np.arange(
                    -custom_div_cmap_arg, custom_div_cmap_arg))
        if saveReport:
            if toPrint:
                logger.info('Save heatmap')
            plt.savefig(
                os.path.join(output_directory, 'Fig_heatmap_ordered'+img_ext),
                transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

    #########################################
    # SAVE filtered data
    if saveReport:
        # save files
        fname = 'data_processed.csv'
        f = os.path.join(output_directory, fname)
        if toPrint:
            logger.info('-save ordered data: '+f)
        data.to_csv(f, sep='\t', header=True, index=True)
