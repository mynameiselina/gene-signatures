# custom imports
from omics_processing.io import (
    set_directory, load_clinical, 
    parse_arg_type, set_path
)
from omics_processing.remove_duplicates import (
    remove_andSave_duplicates
)
from gene_signatures.core import (
    custom_div_cmap,
    get_chr_ticks,
    boxplot,
    which_x_toPrint,
    set_heatmap_size,
    set_cbar_ticks
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


def _split_argument_to_list(
        set_up_kwargs, argument_name,
        asPath, MainDataDir=None, forceType=None):
    # nested lists:
    # if the format of the argument is "string,string;string,string"
    # then the argument_list will be
    # [[string,string],[string,string]] if asPath=False or
    # [MainDataDir/string/string,MainDataDir/string/string] if asPath=True
    #
    # nested lists:
    # if the format of the argument is "string;string"
    # then the argument_list will be
    # [string,string] if asPath=False or
    # [MainDataDir/string/string,MainDataDir/string/string] if asPath=True
    #
    # one level list:
    # if the format of the argument is "string,string,string"
    # then the argument_list will be
    # [string,string,string] if asPath=False or
    # [MainDataDir/string/string/string] if asPath=True
    #
    # one level list:
    # if the format of the argument is "string"
    # then the argument_list will NOT be a list!
    # string if asPath=False or
    # MainDataDir/string/string/string if asPath=True
    #
    # if forceType is not None, instead of string
    # then we get a value of type forceType
    argument = set_up_kwargs.get(
        argument_name, None)
    if argument is None:
        logger.warning(
            'The argument \''+str(argument_name)+'\' is not given!'
        )
        return None
    try:
        # nested lists, get higher level
        if ';' in argument:
            argument_list_nested = argument.rsplit(';')
        else:
            argument_list_nested = [argument]

        argument_list = []
        for single_arg in argument_list_nested:
            # type: string
            formatted_arg = single_arg[:]
            if ',' in single_arg:
                # type: list
                formatted_arg = single_arg.rsplit(',')
            else:
                if forceType is not None:
                    # type: forceType
                    formatted_arg = parse_arg_type(formatted_arg, forceType)
            if asPath:
                # join the path for a single file
                # type: (fpath) string
                formatted_arg = os.path.join(MainDataDir, *formatted_arg)

            # append the formatted_arg from the argument_list_nested
            argument_list.append(formatted_arg)
        # if the argument contained only one level list (not nested)
        if len(argument_list_nested) == 1:
            argument_list = argument_list[0]
    except Exception as ex:
        logger.error(
            'The argument could not be split!\n' +
            argument_name+': '+str(argument))
        logger.error(ex)
        raise

    return argument_list


def combine_features(**set_up_kwargs):
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

    sample_final_id = set_up_kwargs.get('sample_final_id')
    sample_data_ids = set_up_kwargs.get('sample_data_ids')
    if ',' in sample_data_ids:
        sample_data_ids = sample_data_ids.rsplit(',')

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
    MainDataDir = set_path(os.path.join(script_path, '..', 'data'))

    # data input
    file_short_ids = set_up_kwargs.get('file_short_ids', None)
    if ',' in file_short_ids:
        file_short_ids = file_short_ids.rsplit(',')
    else:
        file_short_ids = [file_short_ids]

    data_fpaths = _split_argument_to_list(
        set_up_kwargs, 'files_to_combine_features',
        asPath=True, MainDataDir=MainDataDir)

    # sample info input
    sample_info_directory = set_up_kwargs.get('sample_info_directory')
    if ',' in sample_info_directory:
        sample_info_directory = os.path.join(
            *sample_info_directory.rsplit(','))
    sample_info_directory = os.path.join(
        MainDataDir, sample_info_directory)

    sample_info_fname = set_up_kwargs.get('sample_info_fname')
    if ',' in sample_info_fname:
        sample_info_fname = os.path.join(*sample_info_fname.rsplit(','))
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

    # load info table of samples
    try:
        fpath = os.path.join(sample_info_directory, sample_info_fname)
        sample_info_read_csv_kwargs['col_as_index'] = sample_final_id
        info_table = load_clinical(fpath, **sample_info_read_csv_kwargs)
    except Exception as ex:
        logger.error('Load info table of samples FAILED!')
        logger.error(ex)
        raise

    # load data
    data_dfs = []
    for i, fpath in enumerate(data_fpaths):
        try:
            df = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
            logger.info('loaded data file with shape: '+str(df.shape))
        except Exception as ex:
            logger.error('failed to read data file from: '+str(fpath))
            logger.error(ex)
            raise

        # if datasets have different sample IDs
        # map them to a user defined common one
        if sample_data_ids[i] != sample_final_id:
            # get the two ids from the info_table
            matching_ids = info_table.reset_index()\
                .set_index(sample_data_ids[i])[sample_final_id]
            # add the new id and drop the old one
            # join help with the one-to-one mapping
            df = df.join(matching_ids, how='right')\
                .set_index(sample_final_id, drop=True)

        # add suffix to separate common genes between datasets
        df.columns += "__"+file_short_ids[i]

        data_dfs.append(df)

    # now we join the data features from the multiple datasets
    # on the common samples (inner join)
    data = pd.concat(data_dfs, axis=1, join='inner', sort=False)
    # sort the samples by name
    all_samples = natsorted(data.index.values)
    data = data.loc[all_samples, :]

    # heatmap of combined data (on features)
    _figure_x_size, _figure_y_size, _show_gene_names, _show_sample_names = \
        set_heatmap_size(data)

    plt.figure(figsize=(_figure_x_size, _figure_y_size))
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax,
                     yticklabels=_show_sample_names,
                     xticklabels=_show_gene_names,
                     cmap=cmap_custom, cbar=True)
    plt.xticks(rotation=90)
    plt.title(txt_label)

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

    # save the combined data
    fname = 'data_with_'+sample_final_id+'_id.csv'
    fpath = os.path.join(output_directory, fname)
    logger.info(
        '-save the combined data with different features ' +
        'with shape:\n'+str(data.shape)
    )
    data.to_csv(fpath, sep='\t')


def combine_cohorts(**set_up_kwargs):
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

    # plotting params
    plot_kwargs = set_up_kwargs.get('plot_kwargs', {})
    function_dict = plot_kwargs.get('function_dict', None)
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
    MainDataDir = set_path(os.path.join(script_path, '..', 'data'))

    # data input
    data_fpaths = _split_argument_to_list(
        set_up_kwargs, 'files_to_combine_samples',
        asPath=True, MainDataDir=MainDataDir)

    # data output
    _output_directory = set_up_kwargs.get('output_directory')
    if ',' in _output_directory:
        _output_directory = os.path.join(*_output_directory.rsplit(','))
    _output_directory = os.path.join(MainDataDir, _output_directory)
    output_directory = set_directory(
        os.path.join(_output_directory, reportName)
    )

    # save the set_up_kwargs in the output dir for reproducibility
    fname = 'set_up_kwargs.json'
    f = os.path.join(output_directory, fname)
    if toPrint:
        logger.info(
            '-save set_up_kwargs dictionary for reproducibility in: '+f)
    with open(f, 'w') as fp:
        json.dump(set_up_kwargs, fp, indent=4)

    # sample_info params
    sample_info_kwargs = set_up_kwargs.get('sample_info_kwargs', {})
    save_new_sample_info = False
    if sample_info_kwargs:
        save_new_sample_info = True
        # sample info input
        sample_info_fpaths = _split_argument_to_list(
            sample_info_kwargs, 'sample_info_fpaths',
            asPath=True, MainDataDir=MainDataDir)
        sample_info_read_csv_kwargs = sample_info_kwargs.get(
            'sample_info_read_csv_kwargs', {})
        sample_final_id = _split_argument_to_list(
            sample_info_kwargs, 'sample_final_id', asPath=False)
        sample_info_new_label = _split_argument_to_list(
            sample_info_kwargs, 'sample_info_new_label', asPath=False)
        sample_info_combine_labels = _split_argument_to_list(
            sample_info_kwargs, 'sample_info_combine_labels', asPath=False)
        sample_info_swap_class_label = _split_argument_to_list(
            sample_info_kwargs, 'sample_info_swap_class_label',
            asPath=False)
        # if not isinstance(sample_info_swap_class_label, list):
        #     sample_info_swap_class_label = \
        #         [sample_info_swap_class_label]

        # new sample_info output dir
        new_sample_info_fpath = sample_info_kwargs.get('new_sample_info_fpath')
        if new_sample_info_fpath is None:
            new_sample_info_fpath = _output_directory
        else:
            if ',' in new_sample_info_fpath:
                new_sample_info_fpath = os.path.join(
                    *new_sample_info_fpath.rsplit(','))
                new_sample_info_fpath = os.path.join(
                    MainDataDir, new_sample_info_fpath)

    data_dfs = []
    if save_new_sample_info:
        sample_info_tables = []
    # load info table of samples
    for i, fpath in enumerate(data_fpaths):
        if save_new_sample_info:
            try:
                sample_info_read_csv_kwargs[str(i)]['col_as_index'] = \
                    sample_final_id[i]
                info_table = load_clinical(
                    sample_info_fpaths[i],
                    **sample_info_read_csv_kwargs[str(i)])
            except Exception as ex:
                logger.error('Load info table of samples FAILED!')
                logger.error(ex)
                raise

            if isinstance(sample_info_swap_class_label[i], list):
                _sample_info_swap_class_label_list = \
                    sample_info_swap_class_label[i]
            else:
                _sample_info_swap_class_label_list = \
                    [sample_info_swap_class_label[i]]

            for j in range(len(_sample_info_swap_class_label_list)):
                if _sample_info_swap_class_label_list[j] == '':
                    continue
                logger.warning(
                    'The user requested to swap the ' +
                    str(_sample_info_swap_class_label_list[j]) +
                    ' label in the ' +
                    str(i)+' dataset')
                info_table[_sample_info_swap_class_label_list[j]] = (
                    ~info_table[
                        _sample_info_swap_class_label_list[j]].astype(bool)
                    ).astype(int)

            info_table['dataset'] = i

        # load data
        fpath = data_fpaths[i]
        try:
            df = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
            logger.info('loaded data file with shape: '+str(df.shape))
        except Exception as ex:
            logger.error('failed to read data file from: '+str(fpath))
            logger.error(ex)
            raise

        data_dfs.append(df)
        if save_new_sample_info:
            sample_info_tables.append(info_table)

    # now we join the cohort samples from the multiple datasets
    # on the common features (inner join)
    data = pd.concat(data_dfs, axis=0, join='inner', sort=False)

    # gene info input
    gene_info_fpath = set_up_kwargs.get('gene_info_fpath')
    if gene_info_fpath is not None:
        if ',' in gene_info_fpath:
            gene_info_fpath = os.path.join(
                *gene_info_fpath.rsplit(','))
            gene_info_fpath = os.path.join(
                MainDataDir, gene_info_fpath)

        chr_col = set_up_kwargs.get('chr_col', 'chr_int')
        gene_id_col = set_up_kwargs.get('gene_id_col', 'gene')

        # load gene info
        try:
            genes_positions_table = pd.read_csv(
                gene_info_fpath, sep='\t', header=0, index_col=0)
            # get gene chrom position
            xlabels, xpos = get_chr_ticks(
                genes_positions_table, data,
                id_col='gene', chr_col=chr_col)
        except Exception as ex:
            logger.warning('could not get genes position info')
            logger.warning(ex)
            xlabels, xpos = None, None
    else:
        xlabels, xpos = None, None

    if save_new_sample_info:
        # do the same for the info_tables
        # but keep all collumns (outer join)
        sample_info = pd.concat(
            sample_info_tables, axis=0, join='outer', sort=False)
        sample_info.index.name = 'patientID'
        # create new label name by merging existing labels
        # (when no common label name between cohorts)
        if sample_info_new_label is not None:
            if isinstance(sample_info_new_label, list):
                _sample_info_new_label_list = \
                    sample_info_new_label
                _sample_info_combine_labels_list = \
                    sample_info_combine_labels
            else:
                _sample_info_new_label_list = \
                    [sample_info_new_label]
                _sample_info_combine_labels_list = \
                    [sample_info_combine_labels]
        for l, new_label in enumerate(_sample_info_new_label_list):
            combine_labels = _sample_info_combine_labels_list[l]
            sample_info[new_label] = sample_info[combine_labels].sum(axis=1)
            logger.info(
                'combined labels: '+str(combine_labels) +
                'into the new label: '+str(new_label)
            )

    # sort the samples by name
    all_samples = natsorted(data.index.values)
    data = data.loc[all_samples, :]

    # heatmap of combined data (on samples)

    _figure_x_size, _figure_y_size, _show_gene_names, _show_sample_names = \
        set_heatmap_size(data)

    plt.figure(figsize=(_figure_x_size, _figure_y_size))
    ax = sns.heatmap(data, vmin=vmin, vmax=vmax,
                     yticklabels=_show_sample_names,
                     xticklabels=_show_gene_names,
                     cmap=cmap_custom, cbar=False)
    plt.xticks(rotation=90)
    cbar = ax.figure.colorbar(ax.collections[0])
    set_cbar_ticks(cbar, function_dict, custom_div_cmap_arg)
    plt.title(txt_label)

    if saveReport:
        logger.info('Save heatmap')
        fpath = os.path.join(
            output_directory, 'Fig_heatmap'+img_ext
        )
        plt.savefig(fpath,
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
        data2plot = data.loc[:, gene_order.index].copy()

        # PLOT heatmap after gene ordering
        if toPrint:
            logger.info('Plot heatmap after gene ordering')
        _figure_x_size, _figure_y_size, \
            _show_gene_names, _show_sample_names = \
            set_heatmap_size(data2plot)
        plt.figure(figsize=(_figure_x_size, _figure_y_size))
        ax = sns.heatmap(
            data2plot,
            vmin=vmin, vmax=vmax, xticklabels=_show_gene_names,
            yticklabels=_show_sample_names, cmap=cmap_custom, cbar=False)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, rotation=0)
        cbar = ax.figure.colorbar(ax.collections[0])
        set_cbar_ticks(cbar, function_dict, custom_div_cmap_arg)
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

    # save the combined data
    fname = 'integrated_data.csv'
    fpath = os.path.join(output_directory, fname)
    logger.info(
        '-save the combined data from different cohorts ' +
        'with shape:\n'+str(data.shape)
    )
    data.to_csv(fpath, sep='\t')

    if save_new_sample_info:
        # save the sample_info
        fname = 'integrated_sample_info.csv'
        fpath = os.path.join(new_sample_info_fpath, fname)
        logger.info(
            '-save the combined sample_info from different cohorts ' +
            'with shape:\n'+str(sample_info.shape)
        )
        sample_info.to_csv(fpath, sep='\t')
