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
    edit_names_with_duplicates,
    plot_confusion_matrix
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
from sklearn.metrics import confusion_matrix

# plotting imports
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('white')
sns.set_context('poster')

script_fname = os.path.basename(__file__).rsplit('.')[0]
script_path = os.path.dirname(__file__)
logger = logging.getLogger(__name__)


def _feature_selection_by_classification(
        dat, dat_target, random_state=None, up_thres=0.5, low_thres=0.5):

    if random_state is not None:
        if isinstance(random_state, str) and ',' in random_state:
            _random_state_or = random_state[:]
            random_state = os.path.join(*random_state.rsplit(','))
            try:
                random_state = np.array(random_state).astype(int)
            except Exception as ex:
                logger.warning(
                    "invalid random state values given: "+_random_state_or +
                    "\nrandom_state set to zero\n"+str(ex)
                )
        else:
            random_state = parse_arg_type(random_state, int)
            random_state = np.arange(random_state)
    else:
        random_state = np.arange(10)

    up_thres = parse_arg_type(up_thres, float)
    low_thres = parse_arg_type(low_thres, float)

    logger.info("Running feature selection...")
    dat = dat.copy()
    dat_target = dat_target.copy()

    n_samples = dat.shape[0]
    r_times = random_state.shape[0]
    total_runs = n_samples*r_times

    logger.info(
        "model: svm.LinearSVC with l1 penalty, squared_hinge loss" +
        "and 'balanced' class_weight"
    )

    logger.info(
        "\n\n\'We fit the model n times, where n = number of samples (" +
        str(n_samples)+"), to predict each sample once.\n" +
        "We repeat this r(="+str(r_times)+") times, "
        "each time with a different random state " +
        "and we average each sample\'s prediction. At the end, " +
        "we return the feature coefficients from all (n x r = " +
        str(total_runs)+") models " +
        "and a count of how many correct vs. wrong predictions we had. " +
        "A prediction is set to 1 if >"+str(up_thres) +
        ", 0 if <"+str(low_thres)+" and NA otherwise.\n" +
        "These counts will be evaluated with a bionomial test " +
        "and show how likely it is to get them by chance.\'\n\n"
    )
    n_features = dat.shape[1]
    all_coefs = np.zeros((total_runs, n_features))
    all_y_pred = np.zeros((r_times, n_samples))-1
    logger.info("random_states: "+str(random_state))
    n_runs = 0
    for i, _rs in enumerate(random_state):
        model = svm.LinearSVC(
            penalty='l1', C=1, random_state=_rs,
            loss='squared_hinge', dual=False,
            class_weight='balanced'
        )
        for j in range(dat.shape[0]):
            X = dat.drop(dat.index[j:j+1])
            y = dat_target.drop(dat.index[j:j+1])
            one_sample = dat.iloc[j:j+1, :]
            y_real = dat_target.iloc[j:j+1]

            model.fit(X, y)

            all_coefs[n_runs:n_runs+1, :] = model.coef_[0]
            all_y_pred[i, j] = model.predict(one_sample)

            n_runs += 1

    y_pred = np.mean(all_y_pred, axis=0)
    y_pred[y_pred > up_thres] = 1
    y_pred[y_pred < low_thres] = 0
    y_pred[(low_thres <= y_pred) & (y_pred <= up_thres)] = np.nan
    diffs = np.abs(y_pred - dat_target)
    _save_diffs = diffs.copy()
    missing_pred = np.isnan(diffs)
    diffs[missing_pred] = 1

    wrong = int(diffs.sum())
    correct = int(n_samples - wrong)
    n_missing = int(missing_pred.sum())

    # binomial test
    binom_test_thres = 0.5
    pval = binom_test(correct, n=correct+wrong, p=binom_test_thres)
    printed_results = 'correct = '+str(correct)+', wrong = '+str(wrong) + \
        '\nmissing = '+str(n_missing)+'\npvalue = '+str(pval)
    logger.info(printed_results)
    logger.info("Finished feature selection!")

    # at the end return the model from all samples
    logger.info(
        "Final model to save with all samples and random_state: " +
        str(random_state[0])
    )
    model = svm.LinearSVC(
        penalty='l1', C=1, random_state=random_state[0],
        loss='squared_hinge', dual=False
    )
    X = dat
    y = dat_target
    model.fit(X, y)

    all_coefs = pd.DataFrame(all_coefs, columns=dat.columns.values)
    _save_diffs = pd.Series(_save_diffs, index=y.index)
    _save_diffs.name = "pred_diffs"
    return model, all_coefs, printed_results, \
        (pval, correct, wrong), _save_diffs


def feature_selection(**set_up_kwargs):
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
            class_labels = np.array(class_labels)
    class_values = set_up_kwargs.get('class_values', None)
    if class_values is not None:
        if ',' in class_values:
            class_values = class_values.rsplit(',')
            class_values = np.array(class_values).astype(int)

    # feature_selection_args
    feature_selection_args = set_up_kwargs.get('feature_selection_args', {})
    pval_thres = parse_arg_type(
        feature_selection_args.pop('pval_thres', 0.05),
        float
    )
    topN = parse_arg_type(
        feature_selection_args.pop('topN', 10),
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
    ground_truth.sort_values(inplace=True)
    data = data.reindex(ground_truth.index, axis=0)
    try:
        yticklabels = ground_truth.index.values+',' + \
            ground_truth.values.astype(int).flatten().astype(str)
    except:
        yticklabels = ground_truth.index.values+',' + \
            ground_truth.values.flatten().astype(str)

    # load duplicate genes dictionary
    #  we will need that for the featsel results table we will save later
    if dupl_genes_dict_fpath is not None:
        with open(dupl_genes_dict_fpath, 'r') as fp:
            dupl_genes_dict = json.load(fp)
    else:
        dupl_genes_dict = None

    # Feature Selection
    model, all_coefs, printed_results, \
        (pval, correct, wrong), _sample_pred_diffs = \
        _feature_selection_by_classification(
            data, ground_truth, **feature_selection_args)

    #  save sample prediction scores
    compare_predictions = pd.concat(
        [ground_truth, _sample_pred_diffs], axis=1)
    compare_predictions.loc[:, 'pred_diffs'].fillna(1, inplace=True)
    compare_predictions = compare_predictions.astype(int)
    fname = 'sample_prediciton_scores.csv'
    fpath = os.path.join(output_directory, fname)
    logger.info("-save sample prediction scores in :\n"+fpath)
    compare_predictions.to_csv(
        fpath, sep='\t', header=True, index=True)

    # plot count of correct/wrong predictions per class
    y_maxlim = np.histogram(
        compare_predictions[sample_class_column], bins=2)[0].max()
    axes = compare_predictions.hist(
        by=sample_class_column, column='pred_diffs',
        bins=2, rwidth=0.4, figsize=(10, 6))
    for ax in axes:
        ax.set_ylim(0, y_maxlim+1)
        ax.set_xlim(0, 1)
        ax.set_xticks([0.25, 0.75])
        ax.set_xticklabels(['correct', 'wrong'], rotation=0, fontsize=18)
        ax_title = class_labels[np.where(
            class_values == float(ax.get_title()))[0][0]]+':'+str(ax.get_title())
        ax.set_title(ax_title, fontsize=18)
        plt.suptitle(sample_class_column+' predictions', fontsize=20)
    if saveReport:
        logger.info('Save count plot')
        fpath = os.path.join(
            output_directory, 'Fig_count_plot'+img_ext
        )
        plt.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # plot confusion matrix
    # Compute confusion matrix
    cnf_matrix = confusion_matrix(
        compare_predictions[sample_class_column],
        np.abs(
            compare_predictions[sample_class_column] -
            compare_predictions['pred_diffs']))
    np.set_printoptions(precision=2)
    _classes = [
        class_labels[class_values == 0][0],
        class_labels[class_values == 1][0]]

    # Plot non-normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix, classes=_classes,
        title='Confusion matrix, without normalization')
    if saveReport:
        logger.info('Save count plot')
        fpath = os.path.join(
            output_directory, 'Fig_confusion_matrix'+img_ext
        )
        plt.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # Plot normalized confusion matrix
    plt.figure()
    plot_confusion_matrix(
        cnf_matrix, classes=_classes, normalize=True,
        title='Normalized confusion matrix')
    if saveReport:
        logger.info('Save count plot')
        fpath = os.path.join(
            output_directory, 'Fig_confusion_matrix_normalized'+img_ext
        )
        plt.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # Save to model in the output_directory
    fname = 'joblib_model.pkl'
    fpath = os.path.join(output_directory, fname)
    logger.info('-save model with joblib')
    joblib.dump(model, fpath)

    # get the genes with the nnz coefficients in classification
    featsel_results = pd.DataFrame(index=data.columns.values)
    featsel_results.index.name = 'gene'
    featsel_results['nnz'] = 0
    featsel_results['mean_coef'] = all_coefs.mean(axis=0)
    featsel_results['std_coef'] = all_coefs.std(axis=0)
    nnz_coef_gene_names = (
        np.abs(all_coefs).max(axis=0) > 0
        ).index.values
    featsel_results.loc[nnz_coef_gene_names, 'nnz'] = 1
    n_names = nnz_coef_gene_names.shape[0]

    if dupl_genes_dict is not None:
        featsel_results = edit_names_with_duplicates(
            featsel_results, dupl_genes_dict)

        # change the name of the genes to indicate if they have duplicates
        newgeneNames_data = featsel_results.loc[
            data.columns, 'newGeneName'].values
        newgeneNames_coefs = featsel_results.loc[
            all_coefs.columns, 'newGeneName'].values
        featsel_results.reset_index(inplace=True, drop=False)
        featsel_results.set_index('newGeneName', inplace=True)
        data.columns = newgeneNames_data
        all_coefs.columns = newgeneNames_coefs

        nnz_coef_gene_names = (
            np.abs(all_coefs).max(axis=0) > 0
            ).index.values

    # boxplot of all nnz coefs
    coefs_to_plot = all_coefs.loc[:, nnz_coef_gene_names]
    boxplot(
        coefs_to_plot, coefs_to_plot.shape[1],
        coefs_to_plot.columns.values,
        title=txt_label+" - nnz coef genes",
        txtbox=printed_results, sidespace=2,
        swarm=False, n_names=coefs_to_plot.shape[1]
    )
    if saveReport:
        logger.info('Save boxplot')
        fpath = os.path.join(
            output_directory, 'Fig_boxplot_with_nnz_coefs'+img_ext
        )
        plt.savefig(
            fpath, transparent=True, bbox_inches='tight',
            pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    # heatmap of genes with nnz coefs in classification
    fs_x, fs_y, _show_gene_names, _ = set_heatmap_size(data)
    plt.figure(figsize=(fs_x, fs_y))
    ax = sns.heatmap(data.loc[:, nnz_coef_gene_names], vmin=vmin, vmax=vmax,
                     yticklabels=yticklabels,
                     xticklabels=_show_gene_names,
                     cmap=cmap_custom, cbar=False)
    plt.xticks(rotation=90)
    cbar = ax.figure.colorbar(ax.collections[0])
    set_cbar_ticks(cbar, function_dict, custom_div_cmap_arg)
    plt.title(
        str(n_names)+' genes with nnz coefs in classification: ' +
        class_labels[0]+'['+str(class_values[0])+'] vs. ' +
        class_labels[1]+'['+str(class_values[1])+']'
    )

    if saveReport:
        logger.info('Save heatmap')
        fpath = os.path.join(
            output_directory, 'Fig_heatmap_with_nnz_coefs'+img_ext
        )
        plt.savefig(fpath,
                    transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
        plt.close("all")
    else:
        plt.show()

    if (pval < pval_thres) and (correct > wrong):
        logger.info(
            "selecting genes because bionomial test with pValue < " +
            str(pval_thres)+" and #correct("+str(correct) +
            ") > #wrong("+str(wrong)+") answers"
        )
        featsel_results['abs_mean_coef'] = np.abs(featsel_results['mean_coef'])
        if topN > data.shape[1]:
            topN = data.shape[1]
        featsel_results['top'+str(topN)] = 0
        featsel_results.sort_values(
            by=['abs_mean_coef'], inplace=True, ascending=False)
        selected_gene_names = featsel_results.index.values[:topN]
        featsel_results.loc[selected_gene_names, 'top'+str(topN)] = 1

        # keep only those genes in the data
        data = data.loc[:, selected_gene_names]

        # save this data for future classification
        fname = 'data_features_class.csv'
        fpath = os.path.join(output_directory, fname)
        logger.info("-save data with selected genes\n"+fpath)
        data.to_csv(fpath, sep='\t', header=True, index=True)

        # save as tab-delimited csv file
        fname = 'featsel_results.csv'
        fpath = os.path.join(output_directory, fname)
        logger.info("-save selected genes in :\n"+fpath)
        featsel_results.to_csv(
            fpath, sep='\t', header=True, index=True)

        # save also as excel file
        fname = 'featsel_results.xlsx'
        fpath = os.path.join(output_directory, fname)
        logger.info('-save csv file as excel too')
        writer = pd.ExcelWriter(fpath)
        featsel_results.to_excel(writer)
        writer.save()

        # boxplot of selected coefs
        coefs_to_plot = all_coefs.loc[:, selected_gene_names]
        boxplot(
            coefs_to_plot, coefs_to_plot.shape[1],
            coefs_to_plot.columns.values,
            title=txt_label+" - selected top"+str(topN)+" genes",
            txtbox=printed_results, sidespace=2,
            swarm=False, n_names=coefs_to_plot.shape[1]
        )
        if saveReport:
            logger.info('Save boxplot')
            fpath = os.path.join(
                output_directory, 'Fig_boxplot_with_selected_genes'+img_ext
            )
            plt.savefig(
                fpath, transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

        # heatmap of selected genes
        fs_x, fs_y, _show_gene_names, _ = set_heatmap_size(data)
        plt.figure(figsize=(fs_x, fs_y))
        ax = sns.heatmap(
            data.loc[:, selected_gene_names], vmin=vmin, vmax=vmax,
            yticklabels=yticklabels,
            xticklabels=_show_gene_names,
            cmap=cmap_custom, cbar=False)
        plt.xticks(rotation=90)
        cbar = ax.figure.colorbar(ax.collections[0])
        set_cbar_ticks(cbar, function_dict, custom_div_cmap_arg)
        plt.title(
            'selected top'+str(topN)+' genes: ' +
            class_labels[0]+'['+str(class_values[0])+'] vs. ' +
            class_labels[1]+'['+str(class_values[1])+']'
        )

        if saveReport:
            logger.info('Save heatmap')
            fpath = os.path.join(
                output_directory, 'Fig_heatmap_with_selected_genes'+img_ext
            )
            plt.savefig(fpath,
                        transparent=True, bbox_inches='tight',
                        pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

    else:
        selected_gene_names = None

    if with_swarm:
        # swarmplots
        coefs_to_plot = all_coefs.loc[:, nnz_coef_gene_names]
        boxplot(
            coefs_to_plot, coefs_to_plot.shape[1],
            coefs_to_plot.columns.values,
            title=txt_label+" - nnz coef genes",
            txtbox=printed_results, sidespace=2,
            swarm=True, n_names=coefs_to_plot.shape[1]
        )
        if saveReport:
            logger.info('Save swarmplot')
            fpath = os.path.join(
                output_directory, 'Fig_swarmplot_with_nnz_coefs'+img_ext
            )
            plt.savefig(
                fpath, transparent=True, bbox_inches='tight',
                pad_inches=0.1, frameon=False)
            plt.close("all")
        else:
            plt.show()

        if selected_gene_names is not None:
            coefs_to_plot = all_coefs.loc[:, selected_gene_names]
            boxplot(
                coefs_to_plot, coefs_to_plot.shape[1],
                coefs_to_plot.columns.values,
                title=txt_label+" - nnz coef genes",
                txtbox=printed_results, sidespace=2,
                swarm=True, n_names=coefs_to_plot.shape[1]
            )
            if saveReport:
                logger.info('Save swarmplot')
                fpath = os.path.join(
                    output_directory,
                    'Fig_swarmplot_with_selected_genes'+img_ext
                )
                plt.savefig(
                    fpath, transparent=True, bbox_inches='tight',
                    pad_inches=0.1, frameon=False)
                plt.close("all")
            else:
                plt.show()
