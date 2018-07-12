import os
import numpy as np
import pandas as pd
from natsort import natsorted, index_natsorted
import re
import scipy.stats as stats
from statsmodels.sandbox.stats.multicomp import multipletests
from scipy.stats import binom_test
from sklearn.decomposition import PCA
import logging

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from natsort import natsorted

logger = logging.getLogger(__name__)


def custom_div_cmap(numcolors=11, name='custom_div_cmap',
                    mincol='red', midcol='white', maxcol='blue'):
    # """ Create a custom diverging colormap with three colors
    #
    # Default is blue to white to red with 11 colors.  Colors can be specified
    # in any way understandable by matplotlib.colors.ColorConverter.to_rgb()
    # """

    cmap = LinearSegmentedColormap.from_list(name=name,
                                             colors=[mincol, midcol, maxcol],
                                             N=numcolors)
    return cmap


def get_chr_ticks_deprecated(sorted_gene_table, cnv):

    xticks = np.append(np.arange(1, 22).astype('str'), np.array(['X', 'Y']))

    sorted_gene_table = sorted_gene_table.set_index(['gene']).copy()
    labels = [sorted_gene_table.loc[x]['chr'] for x in cnv.columns]

    chr_size = pd.Series(labels).value_counts()
    chr_size_ar = np.zeros(len(chr_size))

    for cid, counts in enumerate(chr_size):
        try:
            chrInt = int(chr_size.index[cid]) - 1
            chr_size_ar[chrInt] = counts
        except:
            if chr_size.index[cid] == 'X':
                chr_size_ar[22] = counts
            elif chr_size.index[cid] == 'Y':
                chr_size_ar[23] = counts

    chr_size_ar_cum = np.array([int(chr_size_ar[:i+1].cumsum()[-1])
                               for i, c in enumerate(chr_size_ar)])

    return xticks, chr_size_ar_cum


def get_chr_ticks(genes_positions_table, data, id_col='id', chr_col='chr'):
    # make "id" the index for faster lookup
    genes_positions_table = genes_positions_table.set_index([id_col]).copy()
    # get only the labels that exist in the data
    labels = [genes_positions_table.loc[x][chr_col] for x in data.columns]
    # get the unique labels and order them for the xticks
    xticks = np.array(natsorted(set(labels)))
    # count how many genes in the data for each label
    chr_size = pd.Series(labels).value_counts()
    # reorder the labels counts as xticks
    chr_size = chr_size.loc[natsorted(chr_size.index)]
    # the cumulative sum to get the position of the column when each label ends
    chr_endpos = chr_size.cumsum()

    return xticks, chr_endpos


def get_chr_ticks_TCGA(sorted_gene_table, cnv):

    xticks = np.append(np.arange(1, 22).astype('str'), np.array(['X', 'Y']))

    sorted_gene_table = sorted_gene_table.set_index(['gene']).copy()
    labels = [sorted_gene_table.loc[x]['chr'] for x in cnv.columns]

    chr_size = pd.Series(labels).value_counts()
    chr_size_ar = np.zeros(len(chr_size))

    for cid, counts in enumerate(chr_size):
        try:
            chrInt = int(chr_size.index[cid]) - 1
            chr_size_ar[chrInt] = counts
        except:
            if chr_size.index[cid] == 'X':
                chr_size_ar[22] = counts
            elif chr_size.index[cid] == 'Y':
                chr_size_ar[23] = counts

    chr_size_ar_cum = np.array([int(chr_size_ar[:i+1].cumsum()[-1])
                               for i, c in enumerate(chr_size_ar)])

    return xticks, chr_size_ar_cum


def break_yAxis(bottom_axis, top_axis, d=0.005):
    # leave as is
    bottom_axis.spines['bottom'].set_visible(False)
    top_axis.spines['top'].set_visible(False)
    bottom_axis.xaxis.tick_top()
    bottom_axis.tick_params(labeltop='off')
    top_axis.xaxis.tick_bottom()

    kwargs = dict(transform=bottom_axis.transAxes, color='k', clip_on=False)
    bottom_axis.plot((-d, +d), (-d, +d), **kwargs)
    bottom_axis.plot((1-d, 1+d), (-d, +d), **kwargs)
    kwargs.update(transform=top_axis.transAxes)
    top_axis.plot((-d, +d), (1-d, 1+d), **kwargs)
    top_axis.plot((1-d, 1+d), (1-d, 1+d), **kwargs)


def distplot_breakYaxis(x, ymax_bottom, ymax_top, mytitle='',
                        color=None, d=0.005, pad=0, figsize=(10, 5)):
    f, axis = plt.subplots(2, 1, sharex=True, figsize=figsize)
    # choose your plot
    sns.distplot(x.flatten(), ax=axis[0], hist=True, kde=False, color=color)
    sns.distplot(x.flatten(), ax=axis[1], hist=True, kde=False, color=color)

    # set limitis on y axis (play around with threshold)
    axis[0].set_ylim(ymax_top-ymax_bottom, ymax_top)
    axis[0].set_title(mytitle)
    axis[1].set_ylim(0, ymax_bottom)

    # leave as is
    break_yAxis(axis[0], axis[1], d=d)
    plt.tight_layout(pad=pad)


def biplot(dat, ground_truth, pca, pc1=0, pc2=1, n=None, ax=None, isdf=True):
    if not n:
        single_plot = True
        f, ax = plt.subplots(1, 1)
    elif n == 0:
        n = dat.shape[1]
        single_plot = False
        if not ax:
            logger.info('You have to provide the axis ' +
                        'object from the subplot.')
            return None
    elif n == 2:
        single_plot = True
        f, ax = plt.subplots(1, 1)
    else:
        single_plot = False
        if not ax:
            logger.info('You have to provide the axis ' +
                        'object from the subplot.')
            return None

    # project data into PC space

    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[pc1]  # see 'prcomp(my_data)$rotation' in R
    order = abs(xvector).argsort()[::-1]
    xvector = xvector[order]
    yvector = pca.components_[pc2]
    yvector = yvector[order]

    xs = pca.transform(dat)[:, pc1]  # see 'prcomp(my_data)$x' in R
    ys = pca.transform(dat)[:, pc2]

    # visualize projections

    # Note: scale values for arrows and text are a bit inelegant as of now,
    #       so feel free to play around with them

    for i in range(n):
        # arrows project features (ie columns from csv) as vectors onto PC axes
        if isdf:
            ax.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
                     color='k', width=0.0005, head_width=0)
            ax.text(xvector[i]*max(xs), yvector[i]*max(ys),
                    dat.columns.values[order][i], color='k')

            labels = dat.index.values

    for i in range(len(xs)):
        # circles project documents (ie rows from csv) as points onto PC axes
        if ground_truth[i] == 0:
            ax.plot(xs[i], ys[i], 'ro')
            ax.text(xs[i], ys[i], labels[i], color='r')
        else:
            ax.plot(xs[i], ys[i], 'bo')
            ax.text(xs[i], ys[i], labels[i], color='b')

    if single_plot:
        logger.info('test')
        ax.axis([[xs.min()-xs.std(), xs.max()+xs.std(),
                  ys.min()-ys.std(), ys.max()+ys.std()]])
        plt.show()
    else:
        ax.set_xlim([xs.min()-xs.std(), xs.max()+xs.std()])
        ax.set_ylim([ys.min()-ys.std(), ys.max()+ys.std()])

    if isdf:
        return dat.columns.values[order][:n]
    else:
        return None


def biplot2(dat, ground_truth, pca, pc1=0, pc2=1, n=None, ax=None, isdf=True,
            printSampleNames=False, aNum=False):

    if not n:
        single_plot = True
        f, ax = plt.subplots(1, 1)
    elif n == 0:
        n = dat.shape[1]
        single_plot = False
        if not ax:
            logger.info('You have to provide the axis ' +
                        'object from the subplot.')
            return None
    elif n == 2:
        single_plot = True
        f, ax = plt.subplots(1, 1)
    else:
        single_plot = False
        if not ax:
            logger.info('You have to provide the axis ' +
                        'object from the subplot.')
            return None

    # project data into PC space

    # 0,1 denote PC1 and PC2; change values for other PCs
    xvector = pca.components_[pc1]  # see 'prcomp(my_data)$rotation' in R
    order = abs(xvector).argsort()[::-1]
    xvector = xvector[order]
    yvector = pca.components_[pc2]
    yvector = yvector[order]

    xs = pca.transform(dat)[:, pc1]  # see 'prcomp(my_data)$x' in R
    ys = pca.transform(dat)[:, pc2]

    # visualize projections

    # Note: scale values for arrows and text are a bit inelegant as of now,
    #       so feel free to play around with them

    for i in range(n):
        if aNum is None:
            aNum = 0

        # arrows project features (ie columns from csv) as vectors onto PC axes
        if isdf:
            ax.arrow(0, 0, xvector[i]*max(xs), yvector[i]*max(ys),
                     color='k', width=0.0005*min(max(xs), max(ys)),
                     head_width=0)
            gene = dat.columns.values[order][i]
            logger.info(gene)
            # ax.text(xvector[i]*max(xs), yvector[i]*max(ys),
            # 		 gene, color='k', fontsize='small', fontweight='bold')
            offset = (i*aNum*min(max(xs), max(ys)))
            ax.text(xvector[i]*max(xs)+offset, yvector[i]*max(ys)+offset,
                    gene, color='k', fontsize='small', fontweight='bold')

            labels = dat.index.values

    for i in range(len(xs)):
        # circles project documents (ie rows from csv) as points onto PC axes
        if ground_truth[i] == 0:
            ax.plot(xs[i]+(xs[i]*0.15*i), ys[i]+(ys[i]*0.15*i), 'ro')
            if printSampleNames:
                ax.text(xs[i], ys[i], labels[i], color='r')
        else:
            ax.plot(xs[i], ys[i], 'bo')
            if printSampleNames:
                ax.text(xs[i], ys[i], labels[i], color='b')

    if single_plot:
        logger.info('test')
        ax.axis([[xs.min()-xs.std(), xs.max()+xs.std(),
                  ys.min()-ys.std(), ys.max()+ys.std()]])
        plt.show()
    else:
        ax.set_xlim([xs.min()-xs.std(), xs.max()+xs.std()])
        ax.set_ylim([ys.min()-ys.std(), ys.max()+ys.std()])

    if isdf:
        return dat.columns.values[order][:n]
    else:
        return None


def plot_aggr_cnv_TCGA(aggr_ampl, aggr_del, xlabels, xpos, mytitle=''):
    s = aggr_ampl*100
    len_s = len(s)
    plt.figure(figsize=(20, 3))
    plt.xlim(0, len_s)
    markerline1, stemlines1, baseline1 = plt.stem(np.arange(0, len_s), s,
                                                  basefmt='k-')
    plt.setp(markerline1, markerfacecolor='blue', markersize=0)
    plt.setp(stemlines1, linewidth=0.5,
             color=plt.getp(markerline1, 'markerfacecolor'))
    s = aggr_del*100
    markerline2, stemlines2, _ = plt.stem(np.arange(0, len_s), -s,
                                          basefmt='k-')
    plt.setp(markerline2, markerfacecolor='red', markersize=0)
    plt.setp(stemlines2, linewidth=0.5,
             color=plt.getp(markerline2, 'markerfacecolor'))

    plt.xticks(xpos, xlabels, rotation=0)
    plt.xlabel('chromosomes ' +
               '(the number is aligned at the end of the chr region)')
    plt.ylabel('%')

    plt.ylim([-100, 100])

    plt.title(mytitle)
    plt.show()


def plot_aggr_mut(aggr_ampl, aggr_del, xlabels, xpos, mytitle='',
                  printNames=False, font=2, height_space=1,
                  del_space=50, ampl_space=50):

    fig = plt.figure(figsize=(20, 5))
    ax = fig.add_subplot(111)
    plt.axhline(y=0, c='k', linewidth=0.5)
    maxLenGeneName = max(len(max(aggr_ampl.index.values, key=len)),
                         len(max(aggr_del.index.values, key=len)))

    ampl_space = aggr_ampl.shape[0] * 0.0025
    del_space = aggr_del.shape[0] * 0.0025
    ####
    s = aggr_ampl*100
    sMax = s.max()
    y_offset = (sMax*height_space)+maxLenGeneName
    if sMax+y_offset > 100:
        y_offset = 100-sMax
    xs = s.nonzero()[0]
    n = len(xs)
    if n > 0:
        # step = xs.std()*ampl_space
        step = ampl_space
        mid_x = int(-1+n/2)
        new_xs = np.ndarray(n)
        count = 1
        for i in np.arange(mid_x-1, -1, -1):
            new_xs[i] = xs[i]-count*step
            count = count + 1
        new_xs[mid_x] = xs[mid_x]
        count = 1
        for i in np.arange(mid_x+1, n):
            new_xs[i] = xs[i]+count*step
            count = count + 1
    ####
    len_s = len(s)
    ax.set_xlim(-1, len_s)
    ar = np.arange(0, len_s)
    plt.bar(x=np.arange(0, len_s), height=s, width=1, color='b')
    if printNames:
        for i, x in enumerate(xs):
            geneName = aggr_ampl.iloc[x:x+1].index.values[0]
            ax.annotate('%s' % geneName, xy=(new_xs[i], s[x]+y_offset),
                        textcoords='data', fontsize=font, rotation=90,
                        horizontalalignment='center', verticalalignment='top')

    ####
    if (aggr_del > 0).any():
        s = aggr_del*100
        sMax = s.max()
        y_offset = (sMax*height_space)+maxLenGeneName
        if sMax+y_offset > 100:
            y_offset = 100-sMax
        xs = s.nonzero()[0]
        n = len(xs)
        if n > 0:
            # step = xs.std()*del_space
            step = del_space
            mid_x = int(-1+n/2)
            new_xs = np.ndarray(n)
            count = 1
            for i in np.arange(mid_x-1, -1, -1):
                new_xs[i] = xs[i]-count*step
                count = count + 1
            new_xs[mid_x] = xs[mid_x]
            count = 1
            for i in np.arange(mid_x+1, n):
                new_xs[i] = xs[i]+count*step
                count = count + 1
        ####
        plt.bar(x=np.arange(0, len_s), height=-s, width=1, color='r')
        if printNames:
            for i, x in enumerate(xs):
                geneName = aggr_del.iloc[x:x+1].index.values[0]
                ax.annotate('%s' % geneName, xy=(new_xs[i], -s[x]-y_offset),
                            textcoords='data', fontsize=font, rotation=90,
                            horizontalalignment='center',
                            verticalalignment='bottom')

    if xpos is not None:
        plt.xticks(xpos, xlabels, rotation=0)
        plt.xlabel('chromosomes ' +
                   '(the number is aligned at the end of the chr region)')
    elif aggr_ampl.shape[0] < 20:
        plt.xticks(np.arange(aggr_ampl.shape[0]), aggr_ampl.index, rotation=90)
    plt.ylabel('%')
    if (aggr_del > 0).any():
        plt.ylim([-100, 100])
    else:
        plt.ylim([0, 100])
    plt.title(mytitle)


def plot_aggr_var(aggr_mut, aggr_low, xlabels, xpos, mytitle=''):
    s = aggr_mut*100
    len_s = len(s)
    plt.figure(figsize=(10, 2))
    plt.xlim(-1, len_s)

    s = aggr_low*100 + s
    markerline2, stemlines2, _ = plt.stem(np.arange(0, len_s), s, basefmt='k-')
    plt.setp(markerline2, markerfacecolor='red', markersize=2)
    plt.setp(stemlines2, linewidth=2,
             color=plt.getp(markerline2, 'markerfacecolor'))

    s = aggr_mut*100
    markerline1, stemlines1, baseline1 = plt.stem(np.arange(0, len_s), s,
                                                  basefmt='k-')
    plt.setp(markerline1, markerfacecolor='blue', markersize=2)
    plt.setp(stemlines1, linewidth=2,
             color=plt.getp(markerline1, 'markerfacecolor'))

    plt.xticks(xpos, xlabels, rotation=90)
    plt.ylabel('%')

    plt.ylim([0, 200])

    plt.title(mytitle)
    plt.show()


def plot_heatmap(X, y, genes, size=(30, 14), forseDiverge=False):
    # heatmap
    plt.figure(figsize=size)
    target_labels = y.index.values+','+y.values.astype(str)

    if (X.min().min() < 0) or (forseDiverge):
        bar_ticks = np.arange(-5, 5)
        mymin = -4
        mymax = 4
    else:
        bar_ticks = np.arange(0, 9)
        mymin = 0
        mymax = 8
    bwr_custom = custom_div_cmap(9)
    ax = sns.heatmap(X, vmin=mymin, vmax=mymax, yticklabels=target_labels,
                     xticklabels=genes, cmap=bwr_custom,
                     cbar_kws={'ticks': bar_ticks})
    ax.set(xlabel='genes', ylabel='samples')
    plt.show()

    return ax


def plot_heatmap_special(X, sel, X_sel):
    # heatmap
    plt.figure(figsize=(20, X_sel.shape[1]))
    YlOrRd_custom = custom_div_cmap(numcolors=5, name='custom_div_cmap',
                                    mincol='white', midcol='orange',
                                    maxcol='red')
    ax = sns.heatmap(X_sel.T, yticklabels=X.columns[sel.get_support()].values,
                     xticklabels=target_labels, cmap=YlOrRd_custom, cbar=False)

    cbar = ax.figure.colorbar(ax.collections[0])
    myTicks = [0, 1, 2, 3, 4, 5]
    cbar.set_ticks(myTicks)
    cbar.set_ticklabels(pd.Series(myTicks).map(functionImpact_dict_r))

    plt.show()


def boxplot(all_coefs, n, labels, title='', txtbox='',
            sidespace=3, swarm=True, n_names=15):

    sidespace = all_coefs.max() * sidespace
    xpos, xlabels = which_x_toprint(all_coefs, labels, n_names=n_names)

    plt.figure(figsize=(15, 5))
    ax = sns.boxplot(data=all_coefs, color='white', saturation=1, width=0.5,
                     fliersize=0, linewidth=2, whis=1.5, notch=False)
    # iterate over boxes
    for i, box in enumerate(ax.artists):
        if i in xpos:
            box.set_edgecolor('red')
            box.set_facecolor('white')
            # iterate over whiskers and median lines
            for j in range(6*i, 6*(i+1)):
                ax.lines[j].set_color('red')
        else:
            box.set_edgecolor('black')
            box.set_facecolor('white')
            # iterate over whiskers and median lines
            for j in range(6*i, 6*(i+1)):
                ax.lines[j].set_color('black')

    # plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    # plt.setp(ax.lines, color='k')

    if swarm:
        sns.swarmplot(data=all_coefs, color='k', size=3,
                      linewidth=0, ax=ax)

    plt.axhline(y=0, color='k', linewidth=0.2)
    plt.xlim((-1, n))
    plt.xticks(xpos, xlabels, rotation=90)
    plt.title(title)
    lgd = plt.text(0, sidespace, txtbox)


def get_yticklabels(count_yticks):
    if count_yticks < 35:
        yticklabels = True
    elif count_yticks < 60:
        yticklabels = 2
    elif count_yticks < 100:
        yticklabels = 5
    elif count_yticks < 150:
        yticklabels = 10
    elif count_yticks < 200:
        yticklabels = 20
    else:
        yticklabels = False

    return yticklabels


def order_cytoband(cytoband):
    gene_names = cytoband.index
    # use a regular expression to split on p and q arm
    reg = r'(q|p)'
    break_cytoband = pd.DataFrame([list(filter(None, re.split(reg, s)))
                                   for s in cytoband.values],
                                  columns=['chr', 'arm', 'band'],
                                  index=cytoband
                                  )
    # add the gene hugo name
    break_cytoband['gene'] = gene_names
    break_cytoband.reset_index(inplace=True)

    # sort the p and q arms seperately
    # the p arm follows a descending order,
    # whereas the q arm follows an ascending order
    sorted_p = break_cytoband[break_cytoband['arm'] == 'p'
                              ].sort_values(['band', 'gene'],
                                            axis=0, ascending=[False, True],
                                            inplace=False)
    sorted_q = break_cytoband[break_cytoband['arm'] == 'q'
                              ].sort_values(['band', 'gene'],
                                            axis=0, ascending=[True, True],
                                            inplace=False)

    # now merge the 2 tables (first the p arm and then the q arm)
    sorted_cytoband = pd.concat([sorted_p, sorted_q], axis=0, sort=False)

    # natsort the chr THIS WILL COMBINE the arms for each chr and order them
    chr_order = index_natsorted(sorted_cytoband['chr'])
    sorted_cytoband = sorted_cytoband.iloc[chr_order, :]

    sorted_cytoband.reset_index(drop=True, inplace=True)
    sorted_cytoband.index.name = 'order'
    sorted_cytoband.reset_index(drop=False, inplace=True)
    sorted_cytoband.head()

    return sorted_cytoband


def which_x_toprint(df, names, n_names=15):
    if df.shape[1] > 30:
        xmeans = abs(df.mean(axis=0))
        mthres = 0.05
        while (xmeans > mthres).sum() > n_names:
            mthres += 0.01
        xpos = np.arange(df.shape[1])[(xmeans > mthres)]
        xlabels = names[(xmeans > mthres)]
    else:
        xpos = np.arange(df.shape[1])
        xlabels = names

    return xpos, xlabels


def preprocess_oncoscan(onesample, toPrint=False, **kwargs):
    # columns with info about:
    # Chromosome Region, Event, Gene Symbols (in this order!!!)
    if 'keep_columns' not in kwargs.keys():
        logger.error('keep_columns kwarg is missing!')
        raise
    else:
        keep_columns = kwargs['keep_columns']
        if len(keep_columns) > 3:
            logger.error('more than 3 column names are given!\n' +
                         'give columns with info about: ' +
                         'Chromosome Region, Event, Gene Symbols ' +
                         '(in this order!!!)')
            raise

    if 'new_columns' not in kwargs.keys():
        new_columns = ['chr', 'start', 'end', 'id', 'function']

    # choose only the columns: 'Chromosome Region', 'Event', 'Gene Symbols'
    if toPrint:
        logger.info('keep columns: '+str(keep_columns))
    onesample_small = onesample[keep_columns].copy()
    if "'" in onesample_small[keep_columns[0]].iloc[0]:
        if toPrint:
            logger.info("removing the ' character from the Chromosome Region")
        onesample_small[keep_columns[0]] = \
            onesample_small[keep_columns[0]].str.replace("'", "")
    if "-" in onesample_small[keep_columns[0]].iloc[0]:
        if toPrint:
            logger.info("replacing the '-' with ':' character to separate " +
                        "the 'start' and 'end' in the " +
                        "Chromosome Region numbers")
        onesample_small[keep_columns[0]] = \
            onesample_small[keep_columns[0]].str.replace("-", ":")
    # change the gene symbols type (from single string to an array of strings)
    onesample_small['Gene Arrays'] = \
        onesample_small[keep_columns[2]].str.split(', ')
    # remove the row that has NaN in this column
    if onesample_small['Gene Arrays'].isnull().any():
        if toPrint:
            logger.info('remove rows that have NaN gene symbols:')
        temp = onesample_small.shape[0]
        onesample_small.dropna(subset=['Gene Arrays'], inplace=True)
        if toPrint:
            logger.info(' - removed ' +
                        str(temp - onesample_small.shape[0])+' rows')
        if onesample_small.empty:
            logger.warning('after removing rows with no gene symbols, ' +
                           'there are no more CNVs for the patient')
            return onesample_small

    # then reset the index
    onesample_small.reset_index(inplace=True, drop=True)
    # join the event now
    onesample_small['all'] = \
        [map(
            ':'.join,
            zip(np.repeat(onesample_small[keep_columns[0]][row],
                          len(onesample_small['Gene Arrays'][row])),
                onesample_small['Gene Arrays'][row],
                np.repeat(onesample_small[keep_columns[1]][row],
                          len(onesample_small['Gene Arrays'][row]))))
            for row in range(onesample_small.shape[0])]

    # create a new df by flattening the list of genes from each row
    all_data = [y for x in onesample_small['all'].values for y in x]
    onesample_map2genes = pd.DataFrame(all_data, columns=['all'])
    # and now split it again
    onesample_map2genes[new_columns[0]], \
        onesample_map2genes[new_columns[1]], \
        onesample_map2genes[new_columns[2]], \
        onesample_map2genes[new_columns[3]], \
        onesample_map2genes[new_columns[4]] = \
        zip(*onesample_map2genes['all'].str.split(':'))

    # remove duplicates and drop column 'all'
    if toPrint:
        logger.info('remove duplicates with the same value in all columns: ' +
                    str(new_columns))
        logger.info('- '+str(onesample_map2genes.shape[0]) +
                    ' rows before')
    onesample_map2genes = onesample_map2genes.drop_duplicates()
    if toPrint:
        logger.info(' - '+str(onesample_map2genes.shape[0])+' rows after')
    onesample_map2genes = onesample_map2genes.drop(['all'], axis=1)

    # reset index
    onesample_map2genes.reset_index(drop=True, inplace=True)

    if toPrint:
        logger.info('Finished pre-processing successfully.\n')

    return onesample_map2genes


def filter_oncoscan(onesample, toPrint=False, **kwargs):

    if 'col_pValue' not in kwargs.keys():
        logger.error('col_pValue kwarg is missing!')
        raise
    if 'col_probeMedian' not in kwargs.keys():
        logger.error('col_probeMedian kwarg is missing!')
        raise
    if 'col_probeCount' not in kwargs.keys():
        logger.error('col_probeCount kwarg is missing!')
        raise

    if 'pValue_thres' in kwargs.keys():
        pValue_thres = kwargs['pValue_thres']
    else:
        pValue_thres = 0.01
    if 'probeMedian_thres' in kwargs.keys():
        probeMedian_thres = kwargs['probeMedian_thres']
    else:
        probeMedian_thres = 0.3
    if 'probeCount_thres' in kwargs.keys():
        probeCount_thres = kwargs['probeCount_thres']
    else:
        probeCount_thres = 20

    if 'remove_missing_pValues' in kwargs.keys():
        remove_missing_pValues = kwargs['remove_missing_pValues']
    else:
        remove_missing_pValues = False

    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    # keep the rows we will drop
    if remove_missing_pValues:
        r2drop = onesample.index[onesample[kwargs['col_pValue']].isnull()]
        if toPrint:
            logger.info('Filtering out '+str(r2drop.shape[0]) +
                        ' events because the p-Value is missing')
        dropped_rows = dropped_rows.append(onesample.loc[r2drop, :],
                                           sort=False)
        dropped_rows.loc[r2drop, 'reason2drop'] = \
            'filter_'+kwargs['col_pValue']+'_missing'
        # drop the rows
        onesample = onesample.drop(r2drop, axis=0)

    # keep the rows we will drop
    r2drop = onesample.index[onesample[kwargs['col_pValue']] >
                             kwargs['pValue_thres']]
    if toPrint:
        logger.info('Filtering out ' +
                    str(r2drop.shape[0])+' events because p-Value > ' +
                    str(kwargs['pValue_thres']))
    dropped_rows = dropped_rows.append(onesample.loc[r2drop, :], sort=False)
    dropped_rows.loc[r2drop, 'reason2drop'] = \
        'filter_'+kwargs['col_pValue']+'_'+str(kwargs['pValue_thres'])
    # drop the rows
    onesample = onesample.drop(r2drop, axis=0)

    # keep the rows we will drop
    r2drop = onesample.index[abs(onesample[kwargs['col_probeMedian']]) <
                             kwargs['probeMedian_thres']]
    if toPrint:
        logger.info('Filtering out '+str(r2drop.shape[0]) +
                    ' events because probe median < +/-' +
                    str(kwargs['probeMedian_thres']))
    dropped_rows = dropped_rows.append(onesample.loc[r2drop, :], sort=False)
    dropped_rows.loc[r2drop, 'reason2drop'] = \
        'filter_'+kwargs['col_probeMedian']+'_' + \
        str(kwargs['probeMedian_thres'])
    # drop the rows
    onesample = onesample.drop(r2drop, axis=0)

    # keep the rows we will drop
    r2drop = onesample.index[onesample[kwargs['col_probeCount']] <
                             kwargs['probeCount_thres']]
    if toPrint:
        logger.info('Filtering out '+str(r2drop.shape[0]) +
                    ' events because probe count < ' +
                    str(kwargs['probeCount_thres']))
    dropped_rows = dropped_rows.append(onesample.loc[r2drop, :], sort=False)
    dropped_rows.loc[r2drop, 'reason2drop'] = \
        'filter_'+kwargs['col_probeCount']+'_'+str(kwargs['probeCount_thres'])
    # drop the rows
    onesample = onesample.drop(r2drop, axis=0)

    # reset index
    onesample.reset_index(drop=True, inplace=True)
    dropped_rows.reset_index(drop=True, inplace=True)

    if toPrint:
        logger.info('Finished filtering successfully.\n')

    return onesample, dropped_rows


def load_and_process_summary_file(fpaths, info_table, editWith='choose_editor',
                                  toPrint=False, **kwargs):

    if 'comment' in kwargs.keys():
        comment = kwargs['comment']
    else:
        comment = None
    if 'names' in kwargs.keys():
        names = kwargs['names']
    else:
        names = None

    # oncoscan load files from each patient
    data_or = dict()
    data = []
    info_table['oncoscan_events'] = 0
    info_table['oncoscan_events_filt'] = 0
    info_table['genes_with_CNV'] = 0
    info_table['genes_with_CNV_merged'] = 0
    for fpath in fpaths:
        allsamples = pd.read_csv(fpath, sep='\t', header=0,
                                 comment=comment, names=names)
        if 'samples_colname' in kwargs.keys():
            samples_colname = kwargs['samples_colname']
        else:
            samples_colname = allsamples.index.values

        dropped_rows_filt = pd.DataFrame()
        dropped_rows_map = pd.DataFrame()

        for patient_id in natsorted(allsamples[samples_colname].unique()):
            if toPrint:
                logger.info(patient_id)

            data_or[patient_id] = allsamples[allsamples[samples_colname] ==
                                             patient_id].copy()
            onesample = data_or[patient_id].copy()
            info_table.loc[patient_id, 'oncoscan_events'] = onesample.shape[0]
            if toPrint:
                logger.info(str(onesample.shape[0]) +
                            ' oncoscan events for patient ' +
                            str(patient_id))

            if 'filt_kwargs' in kwargs.keys() and kwargs['withFilter']:
                # - pre-process sample - #
                onesample, dropped_rows_filt_pat = \
                    filter_oncoscan(onesample, toPrint=toPrint,
                                    **kwargs['filt_kwargs'])
                if dropped_rows_filt_pat.shape[0] > 0:
                    dropped_rows_filt = pd.concat([dropped_rows_filt,
                                                   dropped_rows_filt_pat],
                                                  axis=0, sort=False)
                info_table.loc[patient_id,
                               'oncoscan_events_filt'] = onesample.shape[0]
                if onesample.empty:
                    logger.warning('after filtering ' +
                                   'there are no more CNVs for patient ' +
                                   str(patient_id))
                    continue
                else:
                    if toPrint:
                        logger.info(str(onesample.shape[0]) +
                                    ' oncoscan events for patient ' +
                                    str(patient_id)+' after filtering')

            if 'preproc_kwargs' in kwargs.keys() and kwargs['withPreprocess']:
                # - pre-process sample - #
                onesample = preprocess_oncoscan(onesample, toPrint=toPrint,
                                                **kwargs['preproc_kwargs'])
                info_table.loc[patient_id,
                               'genes_with_CNV'] = onesample.shape[0]
                if onesample.empty:
                    logger.warning('after pre-processing ' +
                                   'there are no more CNVs for patient ' +
                                   str(patient_id))
                    continue
                else:
                    if toPrint:
                        logger.info(str(onesample.shape[0]) +
                                    ' oncoscan events for patient ' +
                                    str(patient_id) +
                                    ' after after pre-processing')

            np.append(onesample.columns, 'reason2drop')
            if editWith == 'Oncoscan':
                # - edit sample - #
                onesample, dropped_rows_map_pat = \
                    map_oncoscan_to_genes(onesample, patient_id,
                                          toPrint=toPrint,
                                          mergeHow=kwargs['mergeHow'],
                                          removeLOH=kwargs['removeLOH'],
                                          function_dict=kwargs['function_dict']
                                          )
                info_table.loc[patient_id,
                               'genes_with_CNV_merged'] = onesample.shape[0]
                if dropped_rows_map_pat.shape[0] > 0:
                    dropped_rows_map = pd.concat([dropped_rows_map,
                                                  dropped_rows_map_pat],
                                                 axis=0, sort=False)
            else:
                logger.error('unsupported sample editor '+(editWith))
                raise
            data.append(onesample)

    return data, data_or, dropped_rows_filt, dropped_rows_map, info_table


# mergeHow: 'maxAll', 'maxOne', 'freqAll'
def map_oncoscan_to_genes(onesample, sample_name, toPrint=True, removeLOH=True,
                          function_dict=None, mergeHow='maxAll'):

    # for each sample
    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    if 'value' in onesample.columns:
        check_cols = np.delete(onesample.columns.values,
                               np.where(onesample.columns.values == 'value'))
    else:
        check_cols = onesample.columns
    df_isna = onesample.isna()[check_cols]
    if toPrint:
        logger.info('Missing values for each column:\n')
        df_isna_sum = df_isna.sum()
        for _i in range(df_isna_sum.shape[0]):
            logger.info(str(df_isna_sum.index[_i])+'\t' +
                        str(df_isna_sum.iloc[_i]))
    if df_isna.sum().sum() > 0:
        if toPrint:
            logger.info('Remove rows with any missing values in columns:\n' +
                        check_cols)

        # keep the rows we will drop
        r2drop = df_isna[df_isna.any(axis=1)].index
        dropped_rows = dropped_rows.append(onesample.loc[r2drop, :],
                                           sort=False)
        dropped_rows.loc[r2drop, 'reason2drop'] = 'missing_field'

        # drop the rows
        if toPrint:
            logger.info(str(onesample.shape[0])+' rows before')
        onesample.dropna(axis=0, subset=check_cols,  inplace=True)
        if toPrint:
            logger.info(str(onesample.shape[0])+' rows after')

    # remove rows with LOH in FUNCTION !!!!!!!!!!!!!!!!!!
    if removeLOH:
        # keep the rows we will drop
        s_isLOH = (onesample['function'] == 'LOH')
        if s_isLOH.any():
            r2drop = s_isLOH[s_isLOH].index
            dropped_rows = dropped_rows.append(onesample.loc[r2drop, :],
                                               sort=False)
            dropped_rows.loc[r2drop, 'reason2drop'] = 'LOH'

            # drop the rows
            tmp_size = onesample.shape[0]
            onesample.drop(onesample[s_isLOH].index, inplace=True)
            if toPrint:
                if onesample.shape[0] < tmp_size:
                    logger.info('Remove rows with LOH in FUNCTION: ' +
                                str(tmp_size - onesample.shape[0]) +
                                ' rows removed')

    # remove genes that exist in more than one chromosomes
    tmp_size = onesample.shape[0]
    # group by ID and sum over the CHROM
    # (to get all different chrom for one gene)
    chrSum = onesample.groupby(['id'])['chr'].sum()
    # save a dict with only the genes to remove
    # and an array of the diff chroms these gene exist in
    genes_to_remove_dict = \
        {
            chrSum.index[idx]:
            np.unique(np.array(list(filter(None, item.rsplit('chr')))),
                      return_counts=False)
            for idx, item in enumerate(chrSum)
            if len(pd.Series(list(filter(None, item.rsplit('chr')))
                             ).value_counts().values) > 1  # 2) > 1 chroms
            if len(item.rsplit('chr')) > 2   # 1) in more than one positions
        }
    # keep the rows we will drop
    todrop = onesample['id'].isin(genes_to_remove_dict.keys())
    if len(todrop) > 0:
        df2drop = onesample[todrop].copy()
        dropped_rows = dropped_rows.append(df2drop, sort=False)
        dropped_rows.loc[df2drop.index, 'reason2drop'] = 'multiple_chrom'

        # remove those genes
        onesample.drop(onesample[todrop].index, inplace=True)
        if toPrint:
            if onesample.shape[0] < tmp_size:
                logger.info('Remove genes that exist ' +
                            'in multiple chromosomes: ' +
                            str(tmp_size - onesample.shape[0]) +
                            ' rows and ' +
                            str(len(genes_to_remove_dict.keys())) +
                            ' unique gene IDs removed')

    # create a new column with ID and CHR together
    onesample['CHR_ID'] = onesample['chr']+':'+onesample['id']

    # define a dict of FUNCTION values
    # onesample.FUNCTION.unique()
    if function_dict is None:
        function_dict = {
                            'Homozygous Copy Loss': -4,
                            'CN Loss': -2,
                            'CN Gain': 2,
                            'High Copy Gain': 4,
                            'LOH': -1
                        }
    # create a new column with these mapped values
    onesample['FUNC_int'] = onesample['function'].map(function_dict)

    # fist maybe check (and Print) how many chr_id dupl we have
    count_diff = onesample[onesample['CHR_ID'].duplicated(keep='first')
                           ].shape[0]
    if count_diff > 0:
        if toPrint:
            logger.info('Aggregate genes that exist in the same ' +
                        'chromosome multiple times: ' +
                        str(onesample[onesample['CHR_ID'
                                                ].duplicated(keep=False)
                                      ].shape[0]) +
                        'rows aggreagated to' +
                        str(onesample[onesample['CHR_ID'
                                                ].duplicated(keep='first')
                                      ].shape[0]) +
                        'unique rows')
            # these will be aggregated into one row:
            # with function with the highest frequency
            # and overall positional region

    # group by CHR_ID and sum over the FUNCTION
    # (to get all different functions for one gene)
    functionArray = \
        onesample.groupby(['CHR_ID'])['FUNC_int'].apply(
           lambda x: np.append([], x))

    # discard genes with different sign of FUNC_int
    CHR_ID2drop = [functionArray.index[i]
                   for i, item in enumerate(functionArray)
                   if len(np.unique(np.sign(item))) != 1]

    if len(CHR_ID2drop) > 0:
        todrop = onesample['CHR_ID'].isin(CHR_ID2drop)
        df2drop = onesample[todrop].copy()
        dropped_rows = dropped_rows.append(df2drop, sort=False)
        dropped_rows.loc[df2drop.index, 'reason2drop'] = 'ampl_AND_del'

        # remove those genes
        tmp_size = onesample.shape[0]
        onesample.drop(onesample[todrop].index, inplace=True)
        if toPrint:
            if onesample.shape[0] < tmp_size:
                logger.info('Remove genes with amplification AND ' +
                            'deletion values in the same chromosome: ' +
                            str(tmp_size - onesample.shape[0]) +
                            ' rows and '+str(len(CHR_ID2drop)) +
                            ' unique gene IDs removed')

        # group by CHR_ID and sum over the FUNCTION
        # (to get all different functions for one gene)
        functionArray = \
            onesample.groupby(['CHR_ID'])['FUNC_int'].apply(
                lambda x: np.append([], x))

    # group by CHR_ID and concat all the positions
    startPosArray = \
        onesample.groupby(['CHR_ID'])['start'].apply(
            lambda x: np.append([], x))
    endPosArray = \
        onesample.groupby(['CHR_ID'])['end'].apply(
            lambda x: np.append([], x))

    # choose the min start pos
    genes_startPositions = \
        {startPosArray.index[idx]: int(pd.Series(item).min())
         for idx, item in enumerate(startPosArray)}
    # choose the max end pos
    genes_endPositions = \
        {endPosArray.index[idx]: int(pd.Series(item).max())
         for idx, item in enumerate(endPosArray)}

    # mergeHow: 'maxAll', 'maxOne', 'freqAll'
    if mergeHow == 'maxAll':
        if toPrint:
            logger.info(' -Keep the abs max function value per gene ' +
                        'and merge all positions, ' +
                        'with the min start and the max end')
        # Keep the abs max function value per gene
        genes_functions = \
            {functionArray.index[idx]: np.abs(pd.Series(list(item)).unique()
                                              ).argmax()
             for idx, item in enumerate(functionArray)
             if pd.Series(list(item)).unique().size != 0}
        # merge all Pos and take min start and max end
        # and assign the abs max funtion value
        final_dict = dict((':'.join(
                                    [key, str(genes_startPositions[key]),
                                     str(genes_endPositions[key])]),
                           functionArray[key][fidx])
                          for (key, fidx) in genes_functions.items())
    elif mergeHow == 'maxOne':
        if toPrint:
            logger.info(' -Keep the abs max function value per gene ' +
                        'and its position, discard the rest')
        # Keep the abs max function value per gene
        genes_functions = \
            {functionArray.index[idx]: np.abs(pd.Series(list(item)).unique()
                                              ).argmax()
             for idx, item in enumerate(functionArray)
             if pd.Series(list(item)).unique().size != 0}
        # pick the Pos only from abs max funtion value
        final_dict = dict((':'.join(
                                    [key, str(startPosArray[key][fidx]),
                                     str(startPosArray[key][fidx])]),
                           functionArray[key][fidx])
                          for (key, fidx) in genes_functions.items())

    elif mergeHow == 'freqAll':
        if toPrint:
            logger.info(' -Keep the the value with the higher frequency ' +
                        'per gene and merge all positions, ' +
                        'with the min start and the max end')
        # choose the value with the higher frequency
        genes_functions = \
            {functionArray.index[idx]:
             pd.Series(list(item)).value_counts().values.argmax()
             for idx, item in enumerate(functionArray)
             if pd.Series(list(item)).value_counts().size != 0}
        # merge all Pos and take min start and max end
        # and assign the abs max funtion value
        final_dict = dict((':'.join(
                                    [key, str(genes_startPositions[key]),
                                     str(genes_endPositions[key])]),
                           functionArray[key][fidx])
                          for (key, fidx) in genes_functions.items())
    else:
        logger.error('invalid merge option for genes that exist ' +
                     'in the same chromosome multiple times!')
        raise

    # create a pandas Dataframe for one sample
    # with the function integer column and the index broken down in rows
    # one part to merge in one big table of sample (chrid)
    # and the other to define the position
    # (we would need to edit that when merging)
    df = pd.DataFrame.from_dict(final_dict, orient='index')
    df.columns = ['function']
    df.reset_index(inplace=True)  # index to column
    df['chr'], df['id'], df['start'], df['end'] = \
        df['index'].str.split(':', 3).str  # brake the column

    df.drop(['index'], inplace=True, axis=1)  # keep only what we want
    df.set_index(['id'], drop=True, inplace=True)  # re-set the index now
    # put the file/sample name in the column names (because later we merge)
    df.columns = [':'.join([sample_name, name]) for name in df.columns]

    # reset index
    dropped_rows.reset_index(drop=True, inplace=True)
    dropped_rows['sample_name'] = sample_name

    return df, dropped_rows


# choose patient IDs and their order
def choose_samples(ids, dataID, choose_from=None, choose_what=None,
                   sortby=None, **sort_kwargs):
    # choose patients ID
    bool1 = (ids[dataID].isnull() is False)
    # choose a condition to filter patients
    if choose_from is not None:
        bool2 = bool1 & (ids[choose_from] == choose_what)
    else:
        bool2 = bool1
    # sort patients by a column
    if sortby is not None:
        # if type(patient_ids[sortby][0]) is type(''):
        # 	logger.info('str')
        ids = ids[bool2].sort_values(by=sortby, **sort_kwargs)
    else:
        ids = ids[bool2]

    # return the filtered and sorted patient ID (pd.Series)
    return ids[dataID]


def load_clinical(datadir, which_dataID=None, fname=None, **read_csv_kwargs):
    if fname is None:
        fname = 'patient_ids_pandas.txt'
    f = datadir+fname
    patient_ids = pd.read_csv(f, **read_csv_kwargs)

    # check if there are samples missing from the chosen dataID
    # keep only the patients for which we have a sample
    # (for CNV or var depending on our analysis)
    # which_dataID = 'varID'
    # which_dataID = 'cnvID'
    if which_dataID is not None:
        missing_samples = patient_ids[which_dataID].isna()
        if missing_samples.any():
            logger.info(str(missing_samples.sum())+' missing ' +
                        which_dataID+' samples from patient(s): ' +
                        patient_ids['patient'][missing_samples].unique())
            patient_ids.drop(patient_ids.index[missing_samples], axis=0,
                             inplace=True)

    return patient_ids


def get_code_value(patient_ids, valueFrom, codeFrom, code):
    value = patient_ids[valueFrom][patient_ids[codeFrom] == code].unique()
    if len(value) > 1:
        logger.error('problem with patient_ids! Multiple values for ' +
                     codeFrom+' : '+str(value))
        raise
    else:
        value = value[0]
    return value


def format_cnv_for_oncoprint(cnv, function4oncoprint_dict=None,
                             choose_genes=None):
    if function4oncoprint_dict is None:
        function4oncoprint_dict = {
                4: 'AMP',
                3: 'AMP',
                2: 'GAIN',
                1: 'GAIN',
                -1: 'HOMDEL',
                -2: 'HOMDEL',
                -3: 'HOMDEL',
                -4: 'HOMDEL'
            }
    if choose_genes is not None:
        data = cnv[choose_genes].copy()
    else:
        data = cnv.copy()
    rows, cols = data.values.nonzero()

    value = data.iloc[rows, cols]
    sample = data.index[rows]
    gene = data.columns[cols]
    df = pd.DataFrame(data=[sample, gene])

    df = df.T
    df.columns = ['Sample', 'Gene']

    df['Alteration'] = [function4oncoprint_dict[data.iloc[rows[i], cols[i]]]
                        for i in range(len(rows))]
    df['Type'] = 'CNA'

    return df


def get_NexusExpress_diff_analysis(cl1_ampl, cl2_ampl, cl1_del, cl2_del,
                                   with_perc=100, multtest_method='fdr_bh',
                                   multtest_alpha=0.05, min_diff_thres=None,
                                   mytitle=''):

    if len(np.unique([cl1_ampl.shape[0], cl2_ampl.shape[0],
                      cl1_del.shape[0], cl2_del.shape[0]])) != 1:
        logger.error('the groups have different Dimensions!\ncl1_ampl: ' +
                     str(cl1_ampl.shape[0])+'\ncl2_ampl: ' +
                     str(cl2_ampl.shape[0])+'\ncl1_del: ' +
                     str(cl1_del.shape[0])+'\ncl2_del: ' +
                     str(cl2_del.shape[0]))
        raise

    # Multiple test correction with False Discovery Rate and alpha=0.05
    pvalues = pd.Series(index=cl1_ampl.index)
    for gene in cl1_ampl.index:
        # resp vs. non-resp (from primary)
        _, pvalue_ = stats.fisher_exact([[with_perc*cl1_ampl.loc[gene],
                                          with_perc*cl2_ampl.loc[gene]],
                                         [with_perc*cl1_del.loc[gene],
                                          with_perc*cl2_del.loc[gene]]])
        pvalues.loc[gene] = pvalue_

    # reject is true for hypothesis that can be rejected for given alpha
    pvals_reject, pvals_corrected, _, _ = \
        multipletests(pvalues.values, alpha=multtest_alpha,
                      method=multtest_method, returnsorted=False)
    pvals_reject = pd.Series(pvals_reject, index=pvalues.index)
    pvals_corrected = pd.Series(pvals_corrected, index=pvalues.index)
    logger.info(mytitle+' '+multtest_method+': '+str(pvals_reject.sum()) +
                ' sign. diff. genes out of '+str(pvals_reject.shape[0]))

    # compute difference of significantly different genes
    cl1_ampl_new = cl1_ampl.copy()
    cl2_ampl_new = cl2_ampl.copy()
    cl1_del_new = cl1_del.copy()
    cl2_del_new = cl2_del.copy()

    cl1_ampl_new[~pvals_reject] = 0
    cl2_ampl_new[~pvals_reject] = 0
    cl1_del_new[~pvals_reject] = 0
    cl2_del_new[~pvals_reject] = 0

    if min_diff_thres is None:
        min_diff_thres = 0
    gained = abs(cl1_ampl_new - cl2_ampl_new) >= min_diff_thres
    cl1_ampl_new[~gained] = 0
    cl2_ampl_new[~gained] = 0
    logger.info('with '+str(min_diff_thres*with_perc)+'% thres: ' +
                str((cl1_ampl_new != 0).sum()) +
                ' sign. gained genes in code==0')
    logger.info('with '+str(min_diff_thres*with_perc)+'% thres: ' +
                str((cl2_ampl_new != 0).sum()) +
                ' sign. gained genes in code==1')
    logger.info('with '+str(min_diff_thres*with_perc)+'% thres: ' +
                str(gained.sum()) +
                ' sign. gained genes in total')

    deleted = abs(cl1_del_new - cl2_del_new) >= min_diff_thres
    cl1_del_new[~deleted] = 0
    cl2_del_new[~deleted] = 0
    logger.info('with '+str(min_diff_thres*with_perc)+'% thres: ' +
                str((cl1_del_new != 0).sum()) +
                ' sign. deleted genes in code==0')
    logger.info('with '+str(min_diff_thres*with_perc)+'% thres: ' +
                str((cl2_del_new != 0).sum()) +
                ' sign. deleted genes in code==1')
    logger.info('with '+str(min_diff_thres*with_perc)+'% thres: ' +
                str(deleted.sum()) +
                ' sign. deleted genes in total')

    return cl1_ampl_new, cl2_ampl_new, cl1_del_new, cl2_del_new, pvalues, \
        pvals_corrected, pvals_reject, gained, deleted


def PCA_biplots(dat, ground_truth, n_components, random_state=0, title=''):
    pca = PCA(n_components=n_components, svd_solver='full',
              random_state=random_state)
    pca.fit(dat)

    c = pca.n_components_
    logger.info('PCA with '+str(c)+' components')
    logger.info('Explained variance ratio for each component:' +
                str(pca.explained_variance_ratio_))
    logger.info('TOTAL Explained variance ratio:' +
                str(pca.explained_variance_ratio_.sum()))

    f, ax = plt.subplots(c-1, c-1, figsize=(12, 11))
    count = 0
    for i in range(c-1):
        for j in range(c):
            if j <= i:
                ax[j-1, i].axis('off')
            else:
                logger.info(str(i+1)+' vs. '+str(j+1))
                ax[j-1, i].axis('on')
                count = count + 1

                biplot2(dat, ground_truth.values, pca, pc1=i, pc2=j, n=c,
                        ax=ax[j-1, i], isdf=True, aNum=0.1)
                ax[j-1, i].set_xlabel('eigenvector '+str(i+1))
                ax[j-1, i].set_ylabel('eigenvector '+str(j+1))

    plt.suptitle(title)
    # plt.tight_layout()
    # plt.savefig(fname+'_PCA.png')
    # plt.show()

    return pca
