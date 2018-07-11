# custom imports
from omics_processing.io import (
    set_directory, load_clinical
    )
from gene_signatures.core import (
    load_and_process_summary_file,
    custom_div_cmap,
    get_chr_ticks
    )

# basic imports
import os
import sys
import numpy as np
import pandas as pd
import scipy as sp
from natsort import natsorted, index_natsorted

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


def _uniq_chr_per_gene(x, n):
    # f = lambda x, n: x[n] if n < len(x) else np.nan
    return x[n] if n < len(x) else np.nan


def set_up_oncoscan(**set_up_kwargs):
    # initialize script params
    saveReport = set_up_kwargs.get('saveReport', False)
    toPrint = set_up_kwargs.get('toPrint', False)
    reportName = set_up_kwargs.get('reportName', 'test_report_name')
    editWith = set_up_kwargs.get('editWith', 'Oncoscan')
    withFilter = set_up_kwargs.get('withFilter', False)
    withPreprocess = set_up_kwargs.get('withPreprocess', True)
    filt_kwargs = set_up_kwargs.get('filt_kwargs', {})
    preproc_kwargs = set_up_kwargs.get('preproc_kwargs', {})
    txt_label = set_up_kwargs.get('txt_label', 'test_txt_label')
    chr_col = set_up_kwargs.get('chr_col', 'chr_int')
    gene_id_col = set_up_kwargs.get('gene_id_col', 'gene')
    gene_order_dict_fname = set_up_kwargs.get('gene_order_dict_fname', None)
    sample_info_fname = set_up_kwargs.get('sample_info_fname',
                                          '20180704_emca.csv')
    sample_info_table_index_colname = \
        set_up_kwargs.get('sample_info_table_index_colname',
                          'Oncoscan_ID')
    sample_info_table_sortLabels = \
        set_up_kwargs.get('sample_info_table_sortLabels',
                          'TP53_mut5,FOXA1_mut5')
    sample_info_table_sortLabels_list = \
        sample_info_table_sortLabels.rsplit(',')

    # plotting params
    plot_kwargs = set_up_kwargs.get('plot_kwargs', {})
    cmap_custom = plot_kwargs.get('cmap_custom', None)
    vmin = plot_kwargs.get('vmin', None)
    vmax = plot_kwargs.get('vmax', None)

    # initialize directories
    MainDataDir = os.path.join(script_path, '..', 'data')
    input_directory = os.path.join(MainDataDir,
                                   set_up_kwargs.get('input_directory'))
    output_directory = set_directory(
        os.path.join(MainDataDir, set_up_kwargs.get('input_directory'))
        )
    oncoscan_directories = plot_kwargs.get('oncoscan_directories', '')
    oncoscan_directories_list = oncoscan_directories.rsplit(',')
    fpaths = [os.path.join(input_directory, aDir)
              for aDir in oncoscan_directories_list]

    # load info table of samples
    if toPrint:
        print('\nLoad info table of samples')
    fpath = os.path.join(input_directory, sample_info_fname)
    info_table = load_clinical(fpath,
                               col_as_index=sample_info_table_index_colname,
                               **{'na_values': ' '})

    if toPrint:
        print("Missing values for each column:\n", info_table.isna().sum())

    # [optional] load_gene_order_dict(fpath)
    if gene_order_dict_fname is not None:
        fpath = os.path.join(input_directory, gene_order_dict_fname)
        gene_order_dict = load_gene_order_dict(fpath)
    else:
        gene_order_dict = None

    #########################################
    # load files from each patient
    if toPrint:
        print("\n", txt_label, ": load files from all patients\n")

    pat_data_list, pat_data_or_dict, dropped_rows_filt, \
        dropped_rows_map, info_table = \
        load_and_process_summary_file(fpaths, info_table,
                                      editWith=editWith,
                                      toPrint=toPrint,
                                      **set_up_kwargs)

    if (dropped_rows_filt.shape[0] > 0) and (saveReport):
        f_new = 'allsamples__dropped_rows_filt.txt'
        if toPrint:
            print("-save dropped rows from filtering in: ", f_new)
        dropped_rows_filt.to_csv(outDir+f_new, sep='\t',
                                 header=True, index=True)

    if (dropped_rows_map.shape[0] > 0) and (saveReport):
        f_new = 'allsamples__dropped_rows_map.txt'
        if toPrint:
            print("-save dropped rows from mapping oncoscan to genes in: ",
                  f_new)
        dropped_rows_map.to_csv(outDir+f_new, sep='\t',
                                header=True, index=True)

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
        print("\nConcantanate all", editWith,
              "samples in 2 tables (with position, only values)\n")
    # samples in rows, genes in columns
    table_withPos = pd.concat(pat_data_list, join='outer',
                              axis=1, sort=False).T
    if len(remove_patients) > 0:
        if toPrint:
            print("Removing", len(remove_patients),
                  "patients from the analysis, according to user preferances:",
                  remove_patients)
        for patient in remove_patients:
            rows2remove = table_withPos.index.values[
                table_withPos.index.str.contains(remove_patients[0])]
            table_withPos.drop(rows2remove, axis=0, inplace=True)

    # CLEAN THE data FROM ALL SAMPLES
    # extract the start, end and chrom info from the table
    # and keep only the functions values
    start_table = \
        table_withPos[table_withPos.index.str.contains("start")].copy()
    end_table = table_withPos[table_withPos.index.str.contains("end")].copy()
    chr_table = table_withPos[table_withPos.index.str.contains("chr")].copy()
    table = table_withPos.drop(np.concatenate([start_table.index,
                                               end_table.index,
                                               chr_table.index],
                                              axis=0), axis=0)
    if toPrint:
        print("Dimensions of table (samples,genes):", table.shape)
    table.index = [index_name.rsplit(':')[0]
                   for index_name in table.index]
    start_table.index = [index_name.rsplit(':')[0]
                         for index_name in start_table.index]
    end_table.index = [index_name.rsplit(':')[0]
                       for index_name in end_table.index]
    chr_table.index = [index_name.rsplit(':')[0]
                       for index_name in chr_table.index]

    # remove genes that exist in multiple chromosomes across samples
    ll = [list(chr_table[col].dropna().unique()) for col in chr_table.columns]
    n, m = max(map(len, ll)), len(ll)
    uniq_chr_per_gene = pd.DataFrame([[_uniq_chr_per_gene(j, i)
                                       for j in ll] for i in range(n)],
                                     columns=chr_table.columns)
    genes2drop = uniq_chr_per_gene.columns[(~uniq_chr_per_gene.isnull()
                                            ).sum() > 1].values
    if toPrint:
        print("Remove", genes2drop.shape[0],
              "genes that exist in multiple chromosomes across samples:\n",
              genes2drop)

    if (genes2drop.shape[0] > 0):
        if saveReport:
            fname = 'chr_table.csv'
            f = outDir+fname
            if toPrint:
                print("-save chromosomes in: ", f)
            chr_table.to_csv(f, sep='\t', header=True, index=True)

            fname = 'chr_table_uniq.csv'
            f = outDir+fname
            if toPrint:
                print("-save unique chromosomes in: ", f)
            uniq_chr_per_gene.to_csv(f, sep='\t', header=True, index=True)

            fname = 'chr_table_uniq_genes2drop.csv'
            f = outDir+fname
            if toPrint:
                print("-save unique chromosomes from genes to drop in: ", f)
            uniq_chr_per_gene.loc[:, genes2drop].to_csv(f, sep='\t',
                                                        header=True,
                                                        index=True)

        start_table.drop(genes2drop, axis=1, inplace=True)
        end_table.drop(genes2drop, axis=1, inplace=True)
        chr_table.drop(genes2drop, axis=1, inplace=True)
        table.drop(genes2drop, axis=1, inplace=True)
        uniq_chr_per_gene.drop(genes2drop, axis=1, inplace=True)
        uniq_chr_per_gene = uniq_chr_per_gene.iloc[0, :]
        if toPrint:
            print("Dimensions of table (samples,genes):", table.shape)
    else:
        uniq_chr_per_gene = uniq_chr_per_gene.iloc[0, :]

    # ORDER THE GENES FROM ALL SAMPLES
    if toPrint:
        print("\nCreate a Dataframe with the genes " +
              "and their genomic positions")
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
    gene_pos['toNatSort'] = [':'.join([str(gene_pos[chr_col][row]),
                                       str(gene_pos['start'][row]),
                                       str(gene_pos['end'][row])])
                             for row in range(gene_pos.shape[0])]
    if toPrint:
        print('Dataframes agree (?): ' +
              str(gene_pos.shape[0] == table.shape[1]))

    # are the genes duplicated ?
    dupl_genes = gene_pos[gene_id_col].duplicated()
    if dupl_genes.any():
        print("ERROR: genes are duplicated, check your data first!")
        print("duplicated genes:", gene_pos[gene_id_col][dupl_genes].values)
        raise()
    else:
        if toPrint:
            print("genes are unique, continue..")

    if toPrint:
        print("\nOrder genes according to genomic position")
    gene_order = index_natsorted(gene_pos['toNatSort'])
    gene_pos = gene_pos.iloc[gene_order, :]
    gene_pos.reset_index(drop=True, inplace=True)
    gene_pos.index.name = 'order'
    gene_pos.reset_index(inplace=True)

    #########################################
    if gene_order_dict is None:
        # CREATE dictionary of gene names and their order
        gene_order_dict = dict((gene_pos[gene_id_col][i], gene_pos['order'][i])
                               for i in range(gene_pos.shape[0]))
    #########################################
    # ORDER the  table
    if toPrint:
        print("\nOrder  table according to genomic position")
    data = pd.DataFrame(table, columns=sorted(gene_order_dict,
                                              key=gene_order_dict.get))

    #########################################
    oncoscan_count_max = info_table['oncoscan_events'].values.max()
    gene_count_max = info_table['genes_with_CNV'].values.max()
    for label in ['oncoscan_events', 'oncoscan_events_filt',
                  'genes_with_CNV', 'genes_with_CNV_merged']:
        if label in info_table.columns:
            # PLOT Abundance of gene data per sample
            if toPrint:
                print("\nPlot "+label+" for each sample\n")
            mutCount = info_table[[label]].copy()
            patient_new_order = info_table.loc[mutCount.index].sort_values(
                by=sample_info_table_sortLabels_list)
            xticklabels = list(zip(patient_new_order.index.values,
                                   info_table.loc[
                                       patient_new_order.index,
                                       sample_info_table_sortLabels_list
                                                  ].values))
            mutCount = mutCount.loc[patient_new_order.index]
            rank = mutCount[label].argsort().argsort().values
            pal = sns.cubehelix_palette(mutCount.shape[0],
                                        reverse=True, dark=.40, light=.95)
            if True:
                plt.figure(figsize=(10, 5))
                g = sns.barplot(np.arange(mutCount.shape[0]), mutCount[label],
                                palette=np.array(pal[::-1])[rank])
                g.set_xticklabels(xticklabels, rotation=90)
                g.set(xlabel='samples', ylabel='count')
                g.set_title('Abundance of '+label+' per sample\n' +
                            str((mutCount[label] <= 0).sum())+' empty samples')
                if 'oncoscan' in label:
                    plt.ylim([-1, oncoscan_count_max])
                else:
                    plt.ylim([-1, gene_count_max])
                if saveReport:
                    print('Save figure')
                    plt.savefig(outDir+'Fig_samples_'+label+'.png',
                                transparent=True, bbox_inches='tight',
                                pad_inches=0.1, frameon=False)
                if not saveReport:
                    plt.show()
                else:
                    plt.close("all")

    #########################################
    # SAVE table w/ and w/o positions
    if saveReport:
        # save table
        fname = 'table_withPos.csv'
        f = outDir+fname
        if toPrint:
            print("-save table in: ", f)
        table_withPos.to_csv(f, sep='\t', header=True, index=True)

    if saveReport:
        # save table
        fname = 'table.csv'
        f = outDir+fname
        if toPrint:
            print("-save table in: ", f)
        table.to_csv(f, sep='\t', header=True, index=True)

    if toPrint:
        print("Dimensions of table (samples,genes):", table.shape)

    #########################################
    # PLOT heatmap before gene ordering
    if toPrint:
        print("\nPlot heatmap before gene ordering")
    if True:
        plt.figure(figsize=(20, 8))
        patient_new_order = info_table.loc[table.index].sort_values(
            by=sample_info_table_sortLabels_list)
        yticklabels = list(zip(patient_new_order.index.values, info_table.loc[
            patient_new_order.index, sample_info_table_sortLabels_list
            ].values))
        ax = sns.heatmap(table.fillna(0).loc[patient_new_order.index],
                         vmin=vmin, vmax=vmax, xticklabels=False,
                         yticklabels=yticklabels, cmap=cmap_custom, cbar=False)
        cbar = ax.figure.colorbar(ax.collections[0])
        if 'VCF' in editWith:
            myTicks = [0, 1, 2, 3, 4, 5]
            cbar.set_ticks(myTicks)
            cbar.set_ticklabels(pd.Series(myTicks).map(functionImpact_dict_r))
        if saveReport:
            if toPrint:
                print('Save heatmap')
            plt.savefig(outDir+'Fig_heatmap.png',
                        transparent=True, bbox_inches='tight',
                        pad_inches=0.1, frameon=False)
        if not saveReport:
            plt.show()
        else:
            plt.close("all")

    #########################################
    # CHECK if there are empty genes and remove them
    is_empty = (data.isnull()).all(axis=0)
    if is_empty.any():
        genes2remove = data.columns[is_empty]
        data.drop(genes2remove, axis=1, inplace=True)
        if toPrint:
            print("remove the following genes because " +
                  "they have no values in the table: ",
                  genes2remove)

    #########################################
    # PLOT  heatmap after gene ordering and cleaning
    if toPrint:
        print("\nPlot heatmap after gene ordering")
    xlabels, xpos = get_chr_ticks(gene_pos, data, id_col=gene_id_col,
                                  chr_col=chr_col)
    if True:
        plt.figure(figsize=(20, 8))
        patient_new_order = info_table.loc[data.index].sort_values(
            by=sample_info_table_sortLabels_list)
        yticklabels = list(zip(patient_new_order.index.values, info_table.loc[
            patient_new_order.index, sample_info_table_sortLabels_list
            ].values))
        ax = sns.heatmap(data.fillna(0).loc[patient_new_order.index],
                         vmin=vmin, vmax=vmax, xticklabels=False,
                         yticklabels=yticklabels, cmap=cmap_custom, cbar=False)
        ax.set_xticks(xpos)
        ax.set_xticklabels(xlabels, rotation=0)
        cbar = ax.figure.colorbar(ax.collections[0])
        myTicks = np.arange(vmin, vmax+2, 1)
        cbar.set_ticks(myTicks)
        if 'VCF' in editWith:
            cbar.set_ticklabels(pd.Series(myTicks).map(functionImpact_dict_r))
        if saveReport:
            if toPrint:
                print('Save heatmap')
            plt.savefig(outDir+'Fig_heatmap_ordered.png',
                        transparent=True, bbox_inches='tight',
                        pad_inches=0.1, frameon=False)
        if not saveReport:
            plt.show()
        else:
            plt.close("all")

    #########################################
    # SAVE ordered table and gene pos info table
    if saveReport:
        # save files
        fname = 'table_ordered.csv'
        f = outDir+fname
        if toPrint:
            print("-save ordered table: ", f)
        data.to_csv(f, sep='\t', header=True, index=True)

        fname = 'genes_info.csv'
        f = outDir+fname
        if toPrint:
            print("-save genes info: ", f)
        gene_pos.to_csv(f, sep='\t', header=True, index=True)