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
    plot_aggr_mut,
    get_NexusExpress_diff_analysis,
    parse_arg_type
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
from distutils.util import strtobool

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

script_fname = os.path.basename(__file__).rsplit('.')[0]
script_path = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def _get_ampl_del_from_data(data):
    data_ampl = (data > 0).sum(axis=0)/float(data.shape[0])
    data_del = (data < 0).sum(axis=0)/float(data.shape[0])

    return data_ampl, data_del


def _plot_oncoscan_frequency_plot(data_ampl, data_del,
                                  select_samples_title, label,
                                  gene_info_fname, xlabels, xpos,
                                  saveReport, img_ext, output_directory):
    # PLOT freq plot
    if gene_info_fname is not None:
        plot_aggr_mut(data_ampl, data_del, xlabels, xpos,
                      mytitle=select_samples_title+': '+label)
    else:
        plot_aggr_mut(data_ampl, data_del, None, None,
                      mytitle=select_samples_title+': '+label)

    if label == '':
        connection_str = '_'
    else:
        connection_str = ''

    if saveReport:
        fpath = os.path.join(output_directory, 'Fig_Freq_' +
                             select_samples_title+connection_str +
                             label+img_ext)
        logger.info('Save FreqPlot as '+img_ext+' in:\n'+fpath)
        plt.savefig(fpath,
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()


def nexus_express(**set_up_kwargs):
    # chose sample set from data
    # function: choose_samples()
    select_samples_from = set_up_kwargs.get('select_samples_from', None)
    select_samples_which = parse_arg_type(
        set_up_kwargs.get('select_samples_which', None),
        int
    )
    select_samples_sort_by = set_up_kwargs.get('select_samples_sort_by',
                                               None)
    if select_samples_sort_by is not None:
        select_samples_sort_by_list = select_samples_sort_by.rsplit(',')
    select_samples_title = set_up_kwargs.get('select_samples_title',
                                             'select_all')
    clinical_label = select_samples_sort_by_list[0]
    class_labels = set_up_kwargs.get('class_labels', None)
    if class_labels is not None:
        if ',' in class_labels:
            class_labels = class_labels.rsplit(',')
    class_values = set_up_kwargs.get('class_values', None)
    if class_values is not None:
        if ',' in class_values:
            class_values = class_values.rsplit(',')
            class_values = np.array(class_values).astype(int)

    # initialize script params
    saveReport = parse_arg_type(
        set_up_kwargs.get('saveReport', False),
        bool
    )
    toPrint = parse_arg_type(
        set_up_kwargs.get('toPrint', False),
        bool
    )
    toPlotFreq = parse_arg_type(
        set_up_kwargs.get('toPlotFreq', True),
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
    if ',' in sample_info_fname:
        sample_info_fname = os.path.join(*sample_info_fname.rsplit(','))
    sample_info_table_index_colname = \
        set_up_kwargs.get('sample_info_table_index_colname',
                          None)
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})
    data_uniq_fname = input_fname.rsplit('.')[0]+'__' + \
        select_samples_title+'__uniq'
    toRemoveDupl = parse_arg_type(
        set_up_kwargs.get('toRemoveDupl', True),
        bool
    )

    # params for diff analysis
    min_diff_thres = parse_arg_type(
        set_up_kwargs.get('min_diff_thres', 0.25),
        float
    )
    multtest_alpha = parse_arg_type(
        set_up_kwargs.get('multtest_alpha', 0.05),
        float
    )
    with_perc = parse_arg_type(
        set_up_kwargs.get('with_perc', 100),
        int
    )
    multtest_method = set_up_kwargs.get('multtest_method', 'fdr_bh')

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

    # dupl_genes input
    dupl_genes_directory = set_up_kwargs.get('dupl_genes_directory')
    dupl_genes_directory = os.path.join(input_directory, dupl_genes_directory)

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

    # load info table of samples
    if toPrint:
        logger.info('Load info table of samples')
    fpath = os.path.join(sample_info_directory, sample_info_fname)
    info_table = load_clinical(fpath, **sample_info_read_csv_kwargs)

    # load processed data
    fpath = os.path.join(input_directory, input_fname)
    data = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
    empty_pat = data.sum(axis=1).isnull()
    if empty_pat.any():
        logger.info('Patients with missing values in all genes: ' +
                    str(data.index[empty_pat]))
    data = data.fillna(0)

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
    pat_labels_title = str(info_table.reset_index().columns.values)

    # keep only these samples from the data
    data = data.loc[info_table.index, :].copy()

    # plot CNV frequencies of all samples
    data_ampl, data_del = _get_ampl_del_from_data(data)
    if toPlotFreq:
        _plot_oncoscan_frequency_plot(
            data_ampl, data_del, select_samples_title, '',
            gene_info_fname, xlabels, xpos,
            saveReport, img_ext, output_directory)

    extra_label = ''
    if toRemoveDupl:
        # keep a copy of the data with duplicate genes
        data_wDupl = data.copy()
        data_wDupl = data_wDupl.fillna(0)
        xlabels_wDupl = xlabels.copy()
        xpos_wDupl = xpos.copy()
        data_ampl_wDupl, data_del_wDupl = data_ampl.copy(), data_del.copy()

        # load data with uniq genes (this will be the default data from now on)
        fpath = os.path.join(dupl_genes_directory,
                             data_uniq_fname+'.txt')
        if not os.path.exists(fpath):
            logger.warning('The data_uniq file does not exist, ' +
                           'the analysis will run on the processed data ' +
                           'only!\nfile path:\n' +
                           fpath)
            toRemoveDupl = False
        else:
            extra_label = '_uniq'
            data = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
            data = data.fillna(0)

            # keep the same samples as before
            data = data.loc[info_table.index, :].copy()

            # get gene chrom position
            if gene_info_fname is not None:
                xlabels, xpos = get_chr_ticks(genes_positions_table,
                                              data, id_col='gene',
                                              chr_col=chr_col)

            # plot CNV frequencies of all samples with uniq genes
            data_ampl, data_del = _get_ampl_del_from_data(data)
            if toPlotFreq:
                _plot_oncoscan_frequency_plot(
                    data_ampl, data_del, select_samples_title,
                    extra_label, gene_info_fname, xlabels, xpos,
                    saveReport, img_ext, output_directory)

            # load duplicate genes dictionary
            #  we will need that for the table we will save later
            fpath = os.path.join(dupl_genes_directory, data_uniq_fname+'.json')
            with open(fpath, 'r') as fp:
                dupl_genes_dict = json.load(fp)

    # separate patient groups and plot their CNV frequencies
    group0 = data.loc[info_table.index[
        info_table[clinical_label] == class_values[0]]].copy()
    group1 = data.loc[info_table.index[
        info_table[clinical_label] == class_values[1]]].copy()

    group0_ampl, group0_del = _get_ampl_del_from_data(group0)
    if toPlotFreq:
        _plot_oncoscan_frequency_plot(
            group0_ampl, group0_del, select_samples_title,
            class_labels[0]+extra_label, gene_info_fname, xlabels, xpos,
            saveReport, img_ext, output_directory)

    group1_ampl, group1_del = _get_ampl_del_from_data(group1)
    if toPlotFreq:
        _plot_oncoscan_frequency_plot(
            group1_ampl, group1_del, select_samples_title,
            class_labels[1]+extra_label, gene_info_fname, xlabels, xpos,
            saveReport, img_ext, output_directory)

    if toRemoveDupl:
        # plot with the duplicate genes too
        group0_wDupl = data_wDupl.loc[info_table.index[
            info_table[clinical_label] == class_values[0]]].copy()
        group1_wDupl = data_wDupl.loc[info_table.index[
            info_table[clinical_label] == class_values[1]]].copy()

        group0_ampl_wDupl, group0_del_wDupl = \
            _get_ampl_del_from_data(group0_wDupl)
        if toPlotFreq:
            _plot_oncoscan_frequency_plot(
                group0_ampl_wDupl, group0_del_wDupl, select_samples_title,
                class_labels[0], gene_info_fname, xlabels_wDupl,
                xpos_wDupl, saveReport, img_ext, output_directory)

        group1_ampl_wDupl, group1_del_wDupl = \
            _get_ampl_del_from_data(group1_wDupl)
        if toPlotFreq:
            _plot_oncoscan_frequency_plot(
                group1_ampl_wDupl, group1_del_wDupl, select_samples_title,
                class_labels[1], gene_info_fname, xlabels_wDupl,
                xpos_wDupl, saveReport, img_ext, output_directory)

    # run the Nexus Express diff analysis
    # select genes with significant p-value (multtest_alpha)
    # after mutliple test correction (multtest_method) and
    # absolute change higher than the defined threshold (min_diff_thres)
    mytitle = select_samples_title+': '+class_labels[0] +\
        '['+str(class_values[0])+'] vs. ' +\
        class_labels[1]+'['+str(class_values[1])+']'
    group0_ampl_new, group1_ampl_new, group0_del_new, group1_del_new, \
        pvals, pvals_corrected, pvals_reject, gained, deleted = \
        get_NexusExpress_diff_analysis(
            group0_ampl, group1_ampl, group0_del, group1_del,
            with_perc=with_perc, multtest_method=multtest_method,
            multtest_alpha=multtest_alpha, min_diff_thres=min_diff_thres,
            mytitle=mytitle
        )

    # create table with all genes
    if gene_info_fname is not None:
        diff_genes = genes_positions_table.set_index(
            ['gene']).loc[data.columns.values][['chr', 'start', 'end']].copy()
    else:
        diff_genes = pd.DataFrame(index=data.columns.values)
        diff_genes.index.name = 'gene'
    diff_genes[class_labels[0]+'_'+clinical_label+'_ampl'
               ] = group0_ampl*with_perc
    diff_genes[class_labels[1]+'_'+clinical_label+'_ampl'
               ] = group1_ampl*with_perc
    diff_genes[class_labels[0]+'_'+clinical_label+'_del'
               ] = group0_del*with_perc
    diff_genes[class_labels[1]+'_'+clinical_label+'_del'
               ] = group1_del*with_perc

    diff_genes['pvals'] = pvals
    diff_genes['pvals_corrected'] = pvals_corrected
    diff_genes['pvals_reject'] = pvals_reject
    diff_genes['gained'] = gained
    diff_genes['ampl_diff'] = np.abs(
        diff_genes[class_labels[0]+'_'+clinical_label+'_ampl'] -
        diff_genes[class_labels[1]+'_'+clinical_label+'_ampl'])
    diff_genes['deleted'] = deleted
    diff_genes['del_diff'] = np.abs(
        diff_genes[class_labels[0]+'_'+clinical_label+'_del'] -
        diff_genes[class_labels[1]+'_'+clinical_label+'_del'])

    # add the dupl_genes column only if there are duplicate genes
    if toRemoveDupl:
        diff_genes['dupl_genes'] = \
            diff_genes.reset_index()['gene'].map(dupl_genes_dict).values

        # save also the positions of these duplicate genes
        diff_genes['newGeneName'] = diff_genes.index.values
        diff_genes.loc[dupl_genes_dict.keys(), 'newGeneName'] += '__wDupl'
        if gene_info_fname is not None:
            diff_genes['aggChrGene'] = None
            diff_genes['aggPos'] = None
            diff_genes['aggChrStart'] = None
            diff_genes['aggChrEnd'] = None

            # for each duplicated gene, aggregate and save
            # the name, start, end, chr values in the table
            for agene in dupl_genes_dict.keys():
                l = [agene]
                # if agene in dupl_genes_dict.keys():
                l.extend(dupl_genes_dict[agene])
                diff_genes.loc[agene, 'aggChrEnd'] = str(natsorted(
                    genes_positions_table.set_index(
                        'gene').loc[l].reset_index().groupby(
                            by=['chr'])['end'].apply(
                                lambda x: list(np.unique(np.append([], x)))
                                ).reset_index().values.tolist()))
                diff_genes.loc[agene, 'aggChrStart'] = str(natsorted(
                    genes_positions_table.set_index(
                        'gene').loc[l].reset_index().groupby(
                            by=['chr'])['start'].apply(
                                lambda x: list(np.unique(np.append([], x)))
                                ).reset_index().values.tolist()))
                diff_genes.loc[agene, 'aggChrGene'] = str(natsorted(
                    genes_positions_table.set_index(
                        'gene').loc[l].reset_index().groupby(
                            by=['chr'])['gene'].apply(
                                lambda x: list(np.unique(np.append([], x)))
                                ).reset_index().values.tolist()))
                aggPos = \
                    genes_positions_table.set_index('gene').loc[l].groupby(
                        by=['chr']).agg(
                            {'start': min, 'end': max}
                            ).reset_index().astype(str).apply(
                                lambda x: ':'.join(x), axis=1).values
                diff_genes.loc[agene, 'aggPos'] = np.apply_along_axis(
                    lambda x: '__'.join(x), 0, natsorted(aggPos))

    # from the above table: select only the selected genes
    # according to the Nexus Express diff analysis
    diff_genes_selected = diff_genes[
        (diff_genes['gained'] > 0) | (diff_genes['deleted'] > 0)].copy()

    # save tables
    if saveReport:
        fname = 'diff_genes_'+select_samples_title+'.csv'
        fpath = os.path.join(output_directory, fname)
        logger.info("-save all diff genes in :\n"+fpath)
        diff_genes.to_csv(fpath, sep='\t', header=True, index=True)

        if diff_genes_selected.shape[0] > 0:
            # keep only those genes in the data
            data = data.loc[:, diff_genes_selected.index]
            # change the name of the genes to indicate if they have duplicates
            if 'newGeneName' in diff_genes_selected.columns.values:
                newgeneNames = diff_genes_selected.loc[
                    data.columns, 'newGeneName'].values
                data.columns = newgeneNames
            # save this data for future classification
            fname = 'data_features_class.csv'
            fpath = os.path.join(output_directory, fname)
            logger.info("-save data with selected diff genes for " +
                        mytitle+" and samples class labels in :\n"+fpath)
            data.to_csv(fpath, sep='\t', header=True, index=True)

            # save as tab-delimited csv file
            fname = 'diff_genes_selected_'+select_samples_title+'.csv'
            fpath = os.path.join(output_directory, fname)
            logger.info("-save selected diff genes for " +
                        mytitle+" in :\n"+fpath)
            diff_genes_selected.to_csv(fpath, sep='\t',
                                       header=True, index=True)

            # save also as excel file
            fname = 'diff_genes_selected_'+select_samples_title+'.xlsx'
            fpath = os.path.join(output_directory, fname)
            logger.info('-save csv file as excel too')
            writer = pd.ExcelWriter(fpath)
            diff_genes_selected.to_excel(
                writer, sheet_name=select_samples_title)
            writer.save()

    # plot CNV frequencies OF SELECTED GENES for each group in comparison
    if toPlotFreq:
        if ((group0_ampl_new != 0).any() or (group0_del_new != 0).any()):
            _plot_oncoscan_frequency_plot(
                group0_ampl_new, group0_del_new, select_samples_title+'_DIFF',
                class_labels[0]+extra_label, gene_info_fname, xlabels, xpos,
                saveReport, img_ext, output_directory)
        if ((group1_ampl_new != 0).any() or (group1_del_new != 0).any()):
            _plot_oncoscan_frequency_plot(
                group1_ampl_new, group1_del_new, select_samples_title+'_DIFF',
                class_labels[1]+extra_label, gene_info_fname, xlabels, xpos,
                saveReport, img_ext, output_directory)

    if toRemoveDupl:
        group0_ampl_new_wDupl = group0_ampl_wDupl.copy()
        group0_ampl_new_wDupl[:] = 0
        group1_ampl_new_wDupl = group1_ampl_wDupl.copy()
        group1_ampl_new_wDupl[:] = 0
        group0_del_new_wDupl = group0_del_wDupl.copy()
        group0_del_new_wDupl[:] = 0
        group1_del_new_wDupl = group1_del_wDupl.copy()
        group1_del_new_wDupl[:] = 0

        list__diff_genes_selected_wDupl = []
        for i in range(diff_genes_selected.shape[0]):
            theGene = diff_genes_selected.index[i]
            genes2edit = [theGene]
            list__diff_genes_selected_wDupl.extend(genes2edit)
            duplgenes_ = diff_genes_selected.loc[theGene]['dupl_genes']
            if duplgenes_ is not np.nan:
                list__diff_genes_selected_wDupl.extend(duplgenes_)
                genes2edit.extend(duplgenes_)
            group0_ampl_new_wDupl.loc[
                genes2edit] = group0_ampl_new.loc[theGene]
            group1_ampl_new_wDupl.loc[
                genes2edit] = group1_ampl_new.loc[theGene]
            group0_del_new_wDupl.loc[
                genes2edit] = group0_del_new.loc[theGene]
            group1_del_new_wDupl.loc[
                genes2edit] = group1_del_new.loc[theGene]

        if toPlotFreq:
            # plot with the duplicate genes too
            if ((group0_ampl_new_wDupl != 0).any() or
                    (group0_del_new_wDupl != 0).any()):
                _plot_oncoscan_frequency_plot(
                    group0_ampl_new_wDupl, group0_del_new_wDupl,
                    select_samples_title+'_DIFF', class_labels[0],
                    gene_info_fname, xlabels_wDupl, xpos_wDupl,
                    saveReport, img_ext, output_directory
                )
            if ((group1_ampl_new_wDupl != 0).any() or
                    (group1_del_new_wDupl != 0).any()):
                _plot_oncoscan_frequency_plot(
                    group1_ampl_new_wDupl, group1_del_new_wDupl,
                    select_samples_title+'_DIFF', class_labels[1],
                    gene_info_fname, xlabels_wDupl, xpos_wDupl,
                    saveReport, img_ext, output_directory
                )

    # PLOT heatmaps of selected features
    if diff_genes_selected.shape[0] > 0:
        # get only the CNVs from the selected genes
        patientNames2plot = pat_labels_txt
        ds_y, ds_x = data.shape
        fs_x = 25 if ds_x > 45 else 15 if ds_x > 30 else 10
        fs_y = 20 if ds_y > 40 else 15 if ds_y > 30 else 10
        plt.figure(figsize=(fs_x, fs_y))
        ax = sns.heatmap(
            data, vmin=vmin, vmax=vmax,
            xticklabels=True, yticklabels=patientNames2plot,
            cmap=cmap_custom, cbar=False)
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
                cbar.set_ticks(np.arange(-custom_div_cmap_arg,
                                         custom_div_cmap_arg))
        plt.title(mytitle)
        if saveReport:
            fpath = os.path.join(output_directory, 'Fig_Heatmap_' +
                                 select_samples_title +
                                 extra_label+img_ext)
            logger.info('Save Heatmap of selected features as '+img_ext +
                        ' in:\n'+fpath)
            plt.savefig(fpath,
                        transparent=True, bbox_inches='tight',
                        pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

        if toRemoveDupl:
            data2plot = data_wDupl[list__diff_genes_selected_wDupl]
            patientNames2plot = pat_labels_txt
            ds_y, ds_x = data2plot.shape
            fs_x = 25 if ds_x > 45 else 15 if ds_x > 30 else 10
            fs_y = 20 if ds_y > 40 else 15 if ds_y > 30 else 10
            plt.figure(figsize=(fs_x, fs_y))
            ax = sns.heatmap(
                data2plot, vmin=vmin, vmax=vmax,
                xticklabels=True, yticklabels=patientNames2plot,
                cmap=cmap_custom)
            ax.set_ylabel(pat_labels_title)
            cbar = ax.figure.colorbar(ax.collections[0])
            if function_dict is not None:
                functionImpact_dict_r = dict(
                    (v, k) for k, v in function_dict.items()
                    )
                myTicks = [0, 1, 2, 3, 4, 5]
                cbar.set_ticks(myTicks)
                cbar.set_ticklabels(
                    pd.Series(myTicks).map(functionImpact_dict_r))
            else:
                if custom_div_cmap_arg is not None:
                    cbar.set_ticks(
                        np.arange(-custom_div_cmap_arg, custom_div_cmap_arg))
            plt.title(mytitle)
            if saveReport:
                fpath = os.path.join(output_directory, 'Fig_Heatmap_' +
                                     select_samples_title +
                                     '_wDupl'+img_ext)
                logger.info('Save Heatmap of selected features as '+img_ext +
                            ' in:\n'+fpath)
                plt.savefig(fpath,
                            transparent=True, bbox_inches='tight',
                            pad_inches=0.1, frameon=False)
                plt.close("all")
            else:
                plt.show()
