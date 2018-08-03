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
    np.random.seed(random_state)
    logger.info("random state = "+str(random_state))
    # model = svm.LinearSVC(
    #     penalty='l1', C=1, random_state=random_state,
    #     loss='squared_hinge', dual=False
    # )
    model = linear_model.LogisticRegression(
        penalty='l1', C=1, random_state=random_state,
        solver='liblinear'
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
    random_state = parse_arg_type(
        set_up_kwargs.get('random_state', 0),
        int
    )
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
    files_to_combine = set_up_kwargs.get('files_to_combine', None)
    try:
        data_fpaths = []
        if ';' in files_to_combine:
            # split fpaths for different files
            files_to_combine_list = files_to_combine.rsplit(';')
        else:
            files_to_combine_list = [files_to_combine]
        for single_file in files_to_combine_list:
            if ',' in single_file:
                #  join the path for a single file
                data_fpaths.append(
                    os.path.join(MainDataDir, *single_file.rsplit(',')))
    except:
        logger.error(
            'No valid file paths were given to get the features!\n' +
            'files_to_combine: '+str(files_to_combine))
        raise

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
    sample_info_table_index_colname = \
        set_up_kwargs.get('sample_info_table_index_colname')
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})

    # data output
    output_directory = set_up_kwargs.get('output_directory')
    if output_directory is None:
        output_directory = set_directory(
            os.path.join(selected_genes_directory, reportName)
        )
    else:
        if ',' in output_directory:
            output_directory = os.path.join(*output_directory.rsplit(','))
        output_directory = set_directory(
            os.path.join(MainDataDir, output_directory, reportName)
        )

    # load info table of samples
    try:
        fpath = os.path.join(sample_info_directory, sample_info_fname)
        info_table = load_clinical(
            fpath, col_as_index=sample_final_id,
            **sample_info_read_csv_kwargs)
    except:
        logger.error('Load info table of samples FAILED!')
        raise

    # get id column for each dataset
    sample_data_ids = ['cnvID', 'varID']
    sample_final_id = 'cnvID'
    select_columns = sample_data_ids.copy()
    select_columns = set(select_columns).difference(set([sample_final_id]))
    select_columns = np.unique(select_columns)

    # load data
    data_dfs = []
    sample_sets = []
    label_bool = []
    label_list = []
    for i, fpath in enumerate(data_fpaths):
        try:
            df = pd.read_csv(fpath, sep='\t', header=0, index_col=0)
        except:
            logger.error('failed to read data file from: '+str(fpath))
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

        data_dfs.append(df)
        sample_sets.append(set(df.index.values))

        if 'class_label' in df.columns.values:
            label_bool.append(True)
            label_list.append(df['class_label'])
        else:
            label_bool.append(False)

    common_samples = list(set.intersection(*sample_sets))
    # common_samples = natsorted(common_samples)
    # common_samples = list(common_samples)[::-1]

    print("\n\n\n"+str(common_samples)+"\n\n\n")

    label_copies = sum(label_bool)

    # to retrieve the ground truth i.e. the samples class label
    # first we check if the class_label column exists in none
    # or more than one of the datasets
    if label_copies == 0:
        logger.warning(
            "The class label is not in any of the datasets, " +
            "retrieving it from user...")

        clinical_label = set_up_kwargs.get('clinical_label', None)
        if clinical_label is None:
            logger.error('no label was selected from samples info table')
            raise

        common_samples_with_label = set.intersection(
            common_samples, info_table.index)
        if common_samples_with_label < common_samples:
            logger.warning(
                "some samples do not have a class label " +
                "and have to be discarded: " +
                common_samples.difference(common_samples_with_label))
            common_samples = common_samples_with_label

        ground_truth = info_table.loc[common_samples, clinical_label].copy()

    elif label_copies > 1:
        logger.warning(
            "There are "+str(label_copies)+" label copies " +
            "of the class label, comparing them first...")
        joined_labels = pd.concat(label_list, axis=1)
        # makes sense only to compare the common_samples that we keep
        joined_labels = joined_labels.loc[common_samples, :]

        if label_copies == 2:
            comparison = (joined_labels[0] == joined_labels[1])
            different_samples = comparison.index.values[(~comparison)]
        else:
            comparison = \
                joined_labels.iloc[:, 1:]\
                .apply(lambda x: np.where(x == joined_labels[0], True, False),
                       axis=0)\
                .add_suffix('_CALC')
            different_samples = comparison.index.values[
                (~comparison).any(axis=1)]

        if different_samples.size > 0:
            logger.warning(
                "The label copies disagree in " +
                str(different_samples.size) + " samples: " +
                str(different_samples) +
                "\n\nKeeping the first copy from file: " +
                str(np.array(data_fpaths)[label_bool][0]))

        # keep the fist copy
        ground_truth = joined_labels[0].copy()

    else:
        # keep the ONLY copy
        ground_truth = label_list[0].copy()

    # now we concat the data features from the multiple datasets
    data = pd.concat(data_dfs, axis=1, sort=False)
    # select only the common_samples
    data = data.loc[common_samples, :]
    # remove the class_label
    data.drop('class_label', axis=1, inplace=True)

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
    nnz_coef_genes = data.columns.values[(abs(all_coefs).max(axis=0) > 0)]
    clasification_results.loc[nnz_coef_genes, 'classification'] = 1
    n_names = nnz_coef_genes.shape[0]

    logger.info('printing the names of the '+str(n_names) +
                ' features with non-zero coefficient in at least ' +
                'one classification run:\n'+str(nnz_coef_genes))

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

    # save only the data from those samples
    # and the classification selected genes
    fname = 'data_from_classification.csv'
    fpath = os.path.join(output_directory, fname)
    logger.info('-save the classification selected genes')
    data.reindex(nnz_coef_genes, axis=1).to_csv(fpath, sep='\t')

    # boxplot
    boxplot(all_coefs, data.shape[1], data.columns.values,
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
    boxplot(all_coefs, data.shape[1], data.columns.values,
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

    # TODO: need to rething the heatmaps
    # for mutliple datasets of different values
    #
    # # heatmap all selected genes
    # xpos, xlabels = which_x_toPrint(
    #     all_coefs, data.columns.values, n_names=n_names)
    # xpos = xpos + 0.5
    # if data.shape[1] < 6:
    #     figsize = (8, 8)
    # elif data.shape[1] < 15:
    #     figsize = (15, 8)
    # else:
    #     figsize = (25, 8)

    # plt.figure(figsize=figsize)
    # ticklabels = ground_truth.index.values+',' + \
    #     ground_truth.values.flatten().astype(str)
    # bwr_custom = custom_div_cmap(5)
    # ax = sns.heatmap(data, vmin=vmin, vmax=vmax,
    #                  yticklabels=ticklabels, xticklabels=False,
    #                  cmap=cmap_custom, cbar=True)
    # plt.xticks(xpos, xlabels, rotation=90)
    # plt.title(title+' diff mutated genes')

    # if saveReport:
    #     logger.info('Save heatmap')
    #     fpath = os.path.join(
    #         output_directory, 'Fig_'+title+'_heatmap'+img_ext
    #     )
    #     plt.savefig(fpath,
    #                 transparent=True, bbox_inches='tight',
    #                 pad_inches=0.1, frameon=False)
    #     plt.close("all")
    # else:
    #     plt.show()

    # # heatmap of genes with nnz coefs in classification
    # _, xlabels = which_x_toPrint(
    #     all_coefs, data.columns.values, n_names=n_names)
    # if data.shape[1] < 6:
    #     figsize = (8, 8)
    # elif data.shape[1] < 15:
    #     figsize = (15, 8)
    # else:
    #     figsize = (25, 8)

    # plt.figure(figsize=figsize)
    # ticklabels = ground_truth.index.values+',' + \
    #     ground_truth.values.flatten().astype(str)
    # ax = sns.heatmap(data.loc[:, xlabels], vmin=vmin, vmax=vmax,
    #                  yticklabels=ticklabels, xticklabels=True,
    #                  cmap=cmap_custom, cbar=True)
    # plt.title(title+' diff mutated genes - '+str(n_names) +
    #           ' selected from classification coefficients')

    # if saveReport:
    #     logger.info('Save heatmap')
    #     fpath = os.path.join(
    #         output_directory, 'Fig_'+title+'_heatmap2'+img_ext
    #     )
    #     plt.savefig(fpath, transparent=True, bbox_inches='tight',
    #                 pad_inches=0.1, frameon=False)
    #     plt.close("all")
    # else:
    #     plt.show()
