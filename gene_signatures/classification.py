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
    set_heatmap_size,
    set_cbar_ticks,
    edit_names_with_duplicates
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
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

script_fname = os.path.basename(__file__).rsplit('.')[0]
script_path = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def _run_classification(
        dat, dat_target, random_state=None, n_splits=10):

    min_class_count = np.unique(dat_target, return_counts=True)[1].min()
    if n_splits is not None:
        if (n_splits > dat.shape[0]) or (n_splits > min_class_count):
            n_splits = min_class_count
    if random_state is not None:
        random_state = parse_arg_type(random_state, int)
    else:
        random_state = 0
    logger.info(
        "model: svm.LinearSVC with l2 penalty and squared_hinge loss" +
        "random_state: "+str(random_state)
    )
    model = svm.LinearSVC(
        penalty='l2', C=1, random_state=random_state,
        loss='squared_hinge', dual=False
    )

    logger.info("Running classification...")
    dat = dat.copy()
    dat_target = dat_target.copy()

    logger.info(
        "model: svm.LinearSVC with l1 penalty and squared_hinge loss"
    )

    X = dat
    y = dat_target
    k_fold = StratifiedKFold(n_splits=n_splits)
    cross_val_scores = []
    split_i = 0
    all_coefs = np.zeros((n_splits, dat.shape[1]))
    y_train_predictions = pd.Series(index=y.index)
    y_train_predictions.name = "train_predictions"
    for train_indices, test_indices in k_fold.split(X, y):
        X_train = dat.iloc[train_indices]
        y_train = dat_target.iloc[train_indices]

        model.fit(X, y)
        all_coefs[split_i:split_i+1, :] = model.coef_[0]

        X_crossval = dat.iloc[test_indices]
        y_crossval = dat_target.iloc[test_indices]
        cross_val_scores.append(model.score(X_crossval, y_crossval))

        y_train_predictions.iloc[test_indices] = model.predict(X_crossval)

        split_i += 1

    X = dat
    y = dat_target
    model.fit(X, y)

    all_coefs = pd.DataFrame(all_coefs, columns=dat.columns.values)

    return model, all_coefs, y_train_predictions, cross_val_scores


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
    sample_class_column = set_up_kwargs.get('sample_class_column', None)
    if sample_class_column is None:
        logger.error("NO class label was defined!")
        raise
    class_labels = set_up_kwargs.get('class_labels', None)
    if class_labels is not None:
        if ',' in class_labels:
            class_labels = class_labels.rsplit(',')
    class_values = set_up_kwargs.get('class_values', None)
    if class_values is not None:
        if ',' in class_values:
            class_values = class_values.rsplit(',')
            class_values = np.array(class_values).astype(int)

    # feature_selection_args
    classification_args = set_up_kwargs.get('classification_args', {})
    split_train_size = classification_args.pop('split_train_size', 20)
    try:
        if '.' in split_train_size:
            split_train_size = parse_arg_type(split_train_size, float)
        else:
            split_train_size = parse_arg_type(split_train_size, int)
    except:
        pass

    split_random_state = parse_arg_type(
        classification_args.pop('split_random_state', 0),
        int
    )

    # plotting params
    plot_kwargs = set_up_kwargs.get('plot_kwargs', {})
    function_dict = plot_kwargs.get('function_dict', None)
    with_swarm = parse_arg_type(
        plot_kwargs.get('with_swarm', False),
        bool
    )
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
    MainDataDir = os.path.join(script_path, '..', 'data')

    # data input
    data_fpath = set_up_kwargs.get('data_fpath')
    if ',' in data_fpath:
        data_fpath = os.path.join(*data_fpath.rsplit(','))
        data_fpath = os.path.join(MainDataDir, data_fpath)

    # sample info input
    sample_info_fpath = set_up_kwargs.get('sample_info_fpath')
    if ',' in sample_info_fpath:
        sample_info_fpath = os.path.join(*sample_info_fpath.rsplit(','))
    sample_info_fpath = os.path.join(MainDataDir, sample_info_fpath)
    sample_info_read_csv_kwargs = set_up_kwargs.get(
        'sample_info_read_csv_kwargs', {})

    # dupl_genes_dict
    dupl_genes_dict_fpath = set_up_kwargs.get('dupl_genes_dict_fpath', None)
    if dupl_genes_dict_fpath is not None:
        if ',' in dupl_genes_dict_fpath:
            dupl_genes_dict_fpath = os.path.join(
                    *dupl_genes_dict_fpath.rsplit(','))
        dupl_genes_dict_fpath = os.path.join(
            MainDataDir, dupl_genes_dict_fpath)

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

    # load data
    try:
        data = pd.read_csv(data_fpath, sep='\t', header=0, index_col=0)
        logger.info('loaded data file with shape: '+str(data.shape))
    except:
        logger.error('failed to read data file from: '+str(data_fpath))
        raise

    # load info table of samples
    try:
        info_table = load_clinical(
            sample_info_fpath, **sample_info_read_csv_kwargs)
    except:
        logger.error('Load info table of samples FAILED!')
        raise

    # set the ground truth
    ground_truth = info_table.loc[data.index, sample_class_column]

    # load duplicate genes dictionary
    #  we will need that for the featsel results table we will save later
    if dupl_genes_dict_fpath is not None:
        with open(dupl_genes_dict_fpath, 'r') as fp:
            dupl_genes_dict = json.load(fp)
    else:
        dupl_genes_dict = None

    # Classification
    if toTrain:
        # choose labels to stratify train_test_split
        if 'dataset' in info_table.columns.values:
            stratify_by = pd.concat(
                [ground_truth, info_table['dataset']], axis=1)
        else:
            stratify_by = ground_truth
        # split data in train and test
        data_train, data_test, y_train, y_test = train_test_split(
                data, ground_truth,
                train_size=split_train_size,
                test_size=None,
                random_state=split_random_state,
                stratify=stratify_by)

        try:
            yticklabels_train = y_train.index.values+',' + \
                y_train.values.astype(int).flatten().astype(str)
        except:
            yticklabels_train = y_train.index.values+',' + \
                y_train.values.flatten().astype(str)

        # train model
        model, all_coefs, y_train_predictions, y_train_scores = \
            _run_classification(
                data_train, y_train, **classification_args)
    else:
        # load model from file
        model = joblib.load(model_fpath)

        # load features of trained model

        data_test = data
        y_test = ground_truth

    try:
        yticklabels_test = y_test.index.values+',' + \
            y_test.values.astype(int).flatten().astype(str)
    except:
        yticklabels_test = y_test.index.values+',' + \
            y_test.values.flatten().astype(str)

    # Test the model
    y_test_score = model.score(data_test, y_test)
    y_test_predictions = model.predict(data_test)
    y_test_predictions = pd.Series(y_test_predictions, index=data_test.index)
    y_test_predictions.name = 'test_predictions'

    if toTrain:
        #################################################
        # plot accuracy scores of the train and test data
        plt.figure(figsize=(10, 6))
        plt.scatter(
            np.arange(len(y_train_scores))+1, y_train_scores, color='black')
        plt.scatter(0, y_test_score, color='red')
        plt.xlim(-1, len(y_train_scores)+1)
        plt.ylim(0, 1)
        plt.xlabel("test and train kfolds")
        plt.ylabel("accuracy scores")
        if saveReport:
            logger.info('Save boxplot')
            fpath = os.path.join(
                output_directory, 'Fig_scatter'+img_ext
            )
            plt.savefig(
                fpath, transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

        # save train sample cross prediction scores
        y_train_all_labels = pd.concat(
            [y_train, y_train_predictions], axis=1)
        fname = 'y_train_all_labels.csv'
        fpath = os.path.join(output_directory, fname)
        logger.info("-save train labels in :\n"+fpath)
        y_train_all_labels.to_csv(
            fpath, sep='\t', header=True, index=True)

        # Save to model in the output_directory
        fname = 'joblib_model.pkl'
        fpath = os.path.join(output_directory, fname)
        logger.info('-save model with joblib')
        joblib.dump(model, fpath)

        # training classification results
        classification_results = pd.DataFrame(index=data.columns.values)
        classification_results.index.name = 'gene'
        classification_results['mean_coef'] = all_coefs.mean(axis=0)
        classification_results['std_coef'] = all_coefs.std(axis=0)

        if dupl_genes_dict is not None:
            classification_results = edit_names_with_duplicates(
                classification_results, dupl_genes_dict)

            # change the name of the genes to indicate if they have duplicates
            newgeneNames_data = classification_results.loc[
                data.columns, 'newGeneName'].values
            newgeneNames_coefs = classification_results.loc[
                all_coefs.columns, 'newGeneName'].values
            classification_results.reset_index(inplace=True, drop=False)
            classification_results.set_index('newGeneName', inplace=True)
            data.columns = newgeneNames_data
            all_coefs.columns = newgeneNames_coefs

        # boxplot of coefs
        coefs_to_plot = all_coefs
        boxplot(
            coefs_to_plot, coefs_to_plot.shape[1],
            coefs_to_plot.columns.values,
            title=txt_label,
            txtbox='', sidespace=2,
            swarm=False, n_names=coefs_to_plot.shape[1]
        )
        if saveReport:
            logger.info('Save boxplot')
            fpath = os.path.join(
                output_directory, 'Fig_boxplot_train'+img_ext
            )
            plt.savefig(
                fpath, transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

        # heatmap of genes in classification - train
        fs_x, fs_y, _show_gene_names, _ = set_heatmap_size(data_train)
        plt.figure(figsize=(fs_x, fs_y))
        ax = sns.heatmap(
            data_train, vmin=vmin, vmax=vmax,
            yticklabels=yticklabels_train,
            xticklabels=_show_gene_names,
            cmap=cmap_custom, cbar=False)
        plt.xticks(rotation=90)
        cbar = ax.figure.colorbar(ax.collections[0])
        set_cbar_ticks(cbar, function_dict, custom_div_cmap_arg)
        plt.title(
            'train data classification: ' +
            class_labels[0]+'['+str(class_values[0])+'] vs. ' +
            class_labels[1]+'['+str(class_values[1])+']'
        )

        if saveReport:
            logger.info('Save heatmap')
            fpath = os.path.join(
                output_directory, 'Fig_heatmap_train'+img_ext
            )
            plt.savefig(fpath,
                        transparent=True, bbox_inches='tight',
                        pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

        # save as tab-delimited csv file
        fname = 'classification_results.csv'
        fpath = os.path.join(output_directory, fname)
        logger.info("-save selected genes in :\n"+fpath)
        classification_results.to_csv(
            fpath, sep='\t', header=True, index=True)

        # save also as excel file
        fname = 'classification_results.xlsx'
        fpath = os.path.join(output_directory, fname)
        logger.info('-save csv file as excel too')
        writer = pd.ExcelWriter(fpath)
        classification_results.to_excel(writer)
        writer.save()

        if with_swarm:
            # swarmplots
            coefs_to_plot = all_coefs.loc[:, nnz_coef_gene_names]
            boxplot(
                coefs_to_plot, coefs_to_plot.shape[1],
                coefs_to_plot.columns.values,
                title=txt_label,
                txtbox='', sidespace=2,
                swarm=True, n_names=coefs_to_plot.shape[1]
            )
            if saveReport:
                logger.info('Save swarmplot')
                fpath = os.path.join(
                    output_directory, 'Fig_swarmplot_train'+img_ext
                )
                plt.savefig(
                    fpath, transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
                plt.close("all")
            else:
                plt.show()
        #################################################

    # save test sample prediction scores
    y_test_all_labels = pd.concat(
        [y_test, y_test_predictions], axis=1)
    fname = 'y_test_all_labels.csv'
    fpath = os.path.join(output_directory, fname)
    logger.info("-save test labels in :\n"+fpath)
    y_test_all_labels.to_csv(
        fpath, sep='\t', header=True, index=True)

    # heatmap of genes in classification - test
    fs_x, fs_y, _show_gene_names, _ = set_heatmap_size(data_test)
    plt.figure(figsize=(fs_x, fs_y))
    ax = sns.heatmap(data_test, vmin=vmin, vmax=vmax,
                     yticklabels=yticklabels_test,
                     xticklabels=_show_gene_names,
                     cmap=cmap_custom, cbar=False)
    plt.xticks(rotation=90)
    cbar = ax.figure.colorbar(ax.collections[0])
    set_cbar_ticks(cbar, function_dict, custom_div_cmap_arg)
    plt.title(
        'test data classification: ' +
        class_labels[0]+'['+str(class_values[0])+'] vs. ' +
        class_labels[1]+'['+str(class_values[1])+']'
    )

    if saveReport:
        logger.info('Save heatmap')
        fpath = os.path.join(
            output_directory, 'Fig_heatmap_test'+img_ext
        )
        plt.savefig(fpath,
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()
