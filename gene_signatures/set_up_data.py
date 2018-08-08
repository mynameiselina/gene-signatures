# custom imports
from omics_processing.io import (
    set_directory, load_clinical
)
from gene_signatures.core import (
    load_and_process_summary_file,
    load_and_process_files,
    custom_div_cmap,
    get_chr_ticks,
    parse_arg_type,
    choose_samples,
    edit_genepanel
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


def _uniq_chr_per_gene(x, n):
    # f = lambda x, n: x[n] if n < len(x) else np.nan
    return x[n] if n < len(x) else np.nan


def set_up_data(**set_up_kwargs):
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

    # load_data_csv_kwargs = set_up_kwargs.get(
    #     'load_data_csv_kwargs', {}
    # )

    editWith = set_up_kwargs.get('editWith', 'Oncoscan')
    if 'VCF' in editWith:
        _edit_kwargs = set_up_kwargs.get('edit_kwargs', {})
        function_dict = _edit_kwargs.get('function_dict', None)

    txt_label = set_up_kwargs.get('txt_label', 'test_txt_label')

    select_genes = set_up_kwargs.get('select_genes', None)
    if select_genes is None or select_genes == "":
        select_genes_list = []
    else:
        select_genes_list = select_genes.rsplit(',')

    chr_col = set_up_kwargs.get('chr_col', 'chr_int')
    gene_id_col = set_up_kwargs.get('gene_id_col', 'gene')
    sample_info_fname = set_up_kwargs.get('sample_info_fname',
                                          '20180704_emca.csv')
    if ',' in sample_info_fname:
        sample_info_fname = os.path.join(*sample_info_fname.rsplit(','))
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {}
    )
    sample_info_table_sortLabels = \
        set_up_kwargs.get('sample_info_table_sortLabels', None)

    sample_info_table_sortLabels_list = \
        sample_info_table_sortLabels.rsplit(',')

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

    # initialize directories
    MainDataDir = os.path.join(script_path, '..', 'data')
    input_directory = set_up_kwargs.get('input_directory')
    if ',' in input_directory:
        input_directory = os.path.join(*input_directory.rsplit(','))
    input_directory = os.path.join(MainDataDir, input_directory)
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

    data_directory = set_up_kwargs.get('data_directory', None)
    if data_directory is None:
        data_directory = input_directory
    else:
        if ',' in data_directory:
            data_directory = os.path.join(*data_directory.rsplit(','))
    data_files = set_up_kwargs.get('data_files', '')
    data_files_list = data_files.rsplit(',')
    if len(data_files_list) > 0:
        fpaths = [os.path.join(input_directory, data_directory, aFile)
                  for aFile in data_files_list]
    else:
        fpaths = os.path.join(input_directory, data_directory)

    # load info table of samples
    if toPrint:
        logger.info('Load info table of samples')
    fpath = os.path.join(input_directory, sample_info_fname)
    info_table = load_clinical(fpath, **sample_info_read_csv_kwargs)

    if toPrint:
        logger.info('Missing values for each column:\n')
        info_table_isna_sum = info_table.isna().sum()
        for _i in range(info_table_isna_sum.shape[0]):
            logger.info(str(info_table_isna_sum.index[_i])+'\t' +
                        str(info_table_isna_sum.iloc[_i]))

    #########################################
    if 'genepanel' in editWith:
        # print('ERROR: undefined scenario!')
        # raise
        edit_kwargs = set_up_kwargs.get('edit_kwargs', {})

        # load data table
        if len(fpaths) > 0:
            logger.error('more than one data files were given!\n'+str(fpaths))
        variants = pd.read_csv(fpaths[0], sep='\t', header=0)

        # EDIT:
        # map function impact to value with function_dict
        # substitute allele frequencies with impact values
        # aggregate rows to unique genes, choose how to merge
        # remove patients
        data = edit_genepanel(variants, **edit_kwargs)
        xlabels, xpos = None, None

        # plot heatmap
        # save formated data

        # (optional) format data to binary
        # plot heatmap
        # save formated data

    else:
        load_files = parse_arg_type(
            set_up_kwargs.get('load_files', False),
            bool
        )
        # load data/files from each patient
        if load_files:
            if toPrint:
                logger.info(txt_label+': load files from all patients\n')

            pat_data_list, pat_data_or_dict, dropped_rows_filter, \
                dropped_rows_process, dropped_rows_edit, info_table = \
                load_and_process_files(
                    fpaths, info_table, **set_up_kwargs)
        else:
            if toPrint:
                logger.info(txt_label+': load data from all patients\n')

            pat_data_list, pat_data_or_dict, dropped_rows_filter, \
                dropped_rows_process, dropped_rows_edit, info_table = \
                load_and_process_summary_file(
                    fpaths, info_table, **set_up_kwargs)

        if (dropped_rows_filter.shape[0] > 0) and (saveReport):
            f_new = 'allsamples__dropped_rows_filter.txt'
            if toPrint:
                logger.info('-save dropped rows from filtering in:\n'+f_new)
            dropped_rows_filter.to_csv(
                os.path.join(output_directory, f_new),
                sep='\t', header=True, index=True)

        if (dropped_rows_process.shape[0] > 0) and (saveReport):
            f_new = 'allsamples__dropped_rows_process.txt'
            if toPrint:
                logger.info('-save dropped rows from processing in:\n' +
                            f_new)
            dropped_rows_process.to_csv(os.path.join(output_directory, f_new),
                                        sep='\t', header=True, index=True)

        if (dropped_rows_edit.shape[0] > 0) and (saveReport):
            f_new = 'allsamples__dropped_rows_edit.txt'
            if toPrint:
                logger.info('-save dropped rows from editing in:\n' +
                            f_new)
            dropped_rows_edit.to_csv(
                os.path.join(output_directory, f_new),
                sep='\t', header=True, index=True)

        # get size of each sample
        # (i.e. abundance of genes with in each sample)
        # and plot it
        counts = []
        sample_labels = []
        for df in pat_data_list:
            counts.append(df.shape[0])
            sample_labels.append(df.columns[0].rsplit(':')[0])

        ##################################################

        # concat all samples in one table and keep union of all genes,
        # then fill NaNs with zero
        if toPrint:
            logger.info('Concantanate all '+editWith +
                        ' samples in 2 tables (with position, only values)\n')
        # samples in rows, genes in columns
        table_withPos = pd.concat(
            pat_data_list, join='outer', axis=1, sort=False).T

        # CLEAN THE data FROM ALL SAMPLES
        # extract the start, end and chrom info from the table
        # and keep only the functions values
        start_table = \
            table_withPos[table_withPos.index.str.contains('start')].copy()
        end_table = table_withPos[
            table_withPos.index.str.contains('end')].copy()
        chr_table = table_withPos[
            table_withPos.index.str.contains('chr')].copy()
        data = table_withPos.drop(
            np.concatenate([start_table.index,
                            end_table.index,
                            chr_table.index],
                           axis=0), axis=0)
        if toPrint:
            logger.info('Dimensions of data (samples,genes): ' +
                        str(data.shape))
        data.index = [
            index_name.rsplit(':')[0]
            for index_name in data.index]
        start_table.index = [
            index_name.rsplit(':')[0]
            for index_name in start_table.index]
        end_table.index = [
            index_name.rsplit(':')[0]
            for index_name in end_table.index]
        chr_table.index = [
            index_name.rsplit(':')[0]
            for index_name in chr_table.index]

        # remove genes that exist in multiple chromosomes across samples
        ll = [list(chr_table[col].dropna().unique())
              for col in chr_table.columns]
        n, m = max(map(len, ll)), len(ll)
        uniq_chr_per_gene = pd.DataFrame(
            [[
                _uniq_chr_per_gene(j, i)
                for j in ll] for i in range(n)],
            columns=chr_table.columns)
        genes2drop = uniq_chr_per_gene.columns[(~uniq_chr_per_gene.isnull()
                                                ).sum() > 1].values
        if toPrint:
            logger.info('Remove '+str(genes2drop.shape[0]) +
                        ' genes that exist in multiple chromosomes ' +
                        'across samples:\n' +
                        str(genes2drop))

        if (genes2drop.shape[0] > 0):
            # if saveReport:
            #     fname = 'chr_table.csv'
            #     f = os.path.join(output_directory, fname)
            #     if toPrint:
            #         logger.info('-save chromosomes in: '+f)
            #     chr_table.to_csv(f, sep='\t', header=True, index=True)

            #     fname = 'chr_table_uniq.csv'
            #     f = os.path.join(output_directory, fname)
            #     if toPrint:
            #         logger.info('-save unique chromosomes in: '+f)
            #     uniq_chr_per_gene.to_csv(
            #           f, sep='\t', header=True, index=True)

            #     fname = 'chr_table_uniq_genes2drop.csv'
            #     f = os.path.join(output_directory, fname)
            #     if toPrint:
            #         logger.info('-save unique chromosomes ' +
            #                     'from genes to drop in: '+f)
            #     uniq_chr_per_gene.loc[:, genes2drop].to_csv(f, sep='\t',
            #                                                 header=True,
            #                                                 index=True)

            start_table.drop(genes2drop, axis=1, inplace=True)
            end_table.drop(genes2drop, axis=1, inplace=True)
            chr_table.drop(genes2drop, axis=1, inplace=True)
            data.drop(genes2drop, axis=1, inplace=True)
            uniq_chr_per_gene.drop(genes2drop, axis=1, inplace=True)
            uniq_chr_per_gene = uniq_chr_per_gene.iloc[0, :].copy()
            if toPrint:
                logger.info('Dimensions of data (samples,genes):' +
                            str(data.shape))
        else:
            uniq_chr_per_gene = uniq_chr_per_gene.iloc[0, :].copy()

        # ORDER THE GENES FROM ALL SAMPLES (SLOW?)
        if toPrint:
            logger.info('Create a Dataframe with the genes ' +
                        'and their genomic positions')
        gene_pos = pd.concat([
            start_table.apply(
                lambda x: pd.to_numeric(x, errors='ignore', downcast='integer')
                ).min().astype(int),
            end_table.apply(
                lambda x: pd.to_numeric(x, errors='ignore', downcast='integer')
                ).max().astype(int),
            uniq_chr_per_gene],
            axis=1, sort=False)
        gene_pos.columns = ['start', 'end', 'chr']
        gene_pos.index.name = gene_id_col
        gene_pos.reset_index(inplace=True)
        gene_pos['chr_gene'] = gene_pos['chr']+':' + gene_pos[gene_id_col]
        gene_pos[chr_col] = gene_pos['chr'].str.split('chr', 2).str[1]
        gene_pos['toNatSort'] = [
            ':'.join([
                str(gene_pos[chr_col][row]),
                str(gene_pos['start'][row]),
                str(gene_pos['end'][row])])
            for row in range(gene_pos.shape[0])]
        if toPrint:
            logger.info('Dataframes agree (?): ' +
                        str(gene_pos.shape[0] == data.shape[1]))

        # are the genes duplicated ?
        dupl_genes = gene_pos[gene_id_col].duplicated()
        if dupl_genes.any():
            logger.error('genes are duplicated, check your data first!')
            logger.info('duplicated genes:' +
                        gene_pos[gene_id_col][dupl_genes].values)
            raise()
        else:
            if toPrint:
                logger.info('gene names are unique, continue..')

        if toPrint:
            logger.info('Order genes according to genomic position')
        gene_order = index_natsorted(gene_pos['toNatSort'])
        gene_pos = gene_pos.iloc[gene_order, :]
        gene_pos.reset_index(drop=True, inplace=True)
        gene_pos.index.name = 'order'
        gene_pos.reset_index(inplace=True)

        #########################################
        # CREATE dictionary of gene names and their order
        gene_order_dict = dict(
            (gene_pos[gene_id_col][i], int(gene_pos['order'][i]))
            for i in range(gene_pos.shape[0]))

        xlabels, xpos = get_chr_ticks(
            gene_pos, data, id_col=gene_id_col, chr_col=chr_col)

        # SAVE ordered table and gene pos info table
        if saveReport:
            fname = 'genes_info.csv'
            f = os.path.join(output_directory, fname)
            if toPrint:
                logger.info('-save genes info: '+f)
            gene_pos.to_csv(f, sep='\t', header=True, index=True)

            fname = 'gene_order_dict.json'
            f = os.path.join(output_directory, fname)
            if toPrint:
                logger.info('-save genes order dictionary: '+f)
            with open(f, 'w') as fp:
                json.dump(gene_order_dict, fp, indent=4)
        #########################################
        for label in [
                'rows_in_sample', 'rows_in_sample_filt',
                'rows_in_sample_processed', 'rows_in_sample_editted']:
            if label in info_table.columns:
                # PLOT Abundance of gene data per sample
                if toPrint:
                    logger.info('Plot '+label+' for each sample')
                mutCount = info_table[[label]].copy()
                patient_new_order = info_table.loc[mutCount.index].sort_values(
                    by=sample_info_table_sortLabels_list)
                xticklabels = list(zip(
                    patient_new_order.index.values,
                    info_table.loc[
                        patient_new_order.index,
                        sample_info_table_sortLabels_list
                        ].values))
                mutCount = mutCount.loc[patient_new_order.index]
                rank = mutCount[label].argsort().argsort().values
                pal = sns.cubehelix_palette(mutCount.shape[0],
                                            reverse=True, dark=.40, light=.95)
                plt.figure(figsize=(10, 5))
                g = sns.barplot(np.arange(mutCount.shape[0]), mutCount[label],
                                palette=np.array(pal[::-1])[rank])
                g.set_xticklabels(xticklabels, rotation=90)
                g.set(xlabel='samples', ylabel='count')
                g.set_title('Abundance of '+label+' per sample: ' +
                            str((mutCount[label] <= 0).sum())+' empty samples')
                if saveReport:
                    logger.info('Save figure')
                    plt.savefig(os.path.join(output_directory, 'Fig_samples_' +
                                label+img_ext),
                                transparent=True, bbox_inches='tight',
                                pad_inches=0.1, frameon=False)
                    plt.close("all")
                else:
                    plt.show()

        #########################################
        # # SAVE data w/ and w/o positions
        # if saveReport:
        #     # save data
        #     fname = 'table_withPos.csv'
        #     f = os.path.join(output_directory, fname)
        #     if toPrint:
        #         logger.info('-save data in: '+f)
        #     table_withPos.to_csv(f, sep='\t', header=True, index=True)

        # if saveReport:
        #     # save data
        #     fname = 'data.csv'
        #     f = os.path.join(output_directory, fname)
        #     if toPrint:
        #         logger.info('-save data in: '+f)
        #     data.to_csv(f, sep='\t', header=True, index=True)

        # if toPrint:
        #     logger.info(
        #       'Dimensions of data (samples,genes):'+str(data.shape))

    #  -- END IF -- #
    #########################################
    # SAVE data and sample_info
    if saveReport:
        # save files
        fname = 'data.csv'
        f = os.path.join(output_directory, fname)
        if toPrint:
            logger.info('-save ordered data: '+f)
        data.to_csv(f, sep='\t', header=True, index=True)

        fname = 'sample_info.csv'
        f = os.path.join(output_directory, fname)
        if toPrint:
            logger.info('-save sample_info: '+f)
        info_table.to_csv(f, sep='\t', header=True, index=True)
