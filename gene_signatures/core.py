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
from distutils.util import strtobool

import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from natsort import natsorted

logger = logging.getLogger(__name__)


def parse_arg_type(arg, type):
    if arg is not None:
        if not isinstance(arg, type):
            if type == bool:
                arg = bool(strtobool(arg))
            else:
                arg = type(arg)
    return arg


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
    # plt.tight_layout(pad=pad)


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

    all_coefs = all_coefs.values.copy()
    sidespace = all_coefs.max() * sidespace
    xpos, xlabels = which_x_toPrint(all_coefs, labels, n_names=n_names)

    xsize = all_coefs.shape[1]
    if xsize > 150:
        figsize_x = 40
        x_font_size = 5
    elif xsize > 30:
        figsize_x = 25
        x_font_size = 10
    else:
        figsize_x = 15
        x_font_size = 20

    plt.figure(figsize=(figsize_x, 5))
    ax = sns.boxplot(data=all_coefs, color='white', saturation=1, width=0.5,
                     fliersize=2, linewidth=2, whis=1.5, notch=False)
    # # iterate over boxes
    # for i, box in enumerate(ax.artists):
    #     if i in xpos:
    #         box.set_edgecolor('red')
    #         # box.set_facecolor('white')
    #         # iterate over whiskers and median lines
    #         for j in range(6*i, 6*(i+1)):
    #             ax.lines[j].set_color('red')
    #     else:
    #         box.set_edgecolor('black')
    #         # box.set_facecolor('white')
    #         # iterate over whiskers and median lines
    #         for j in range(6*i, 6*(i+1)):
    #             ax.lines[j].set_color('black')

    # plt.setp(ax.artists, edgecolor = 'k', facecolor='w')
    # plt.setp(ax.lines, color='k')

    if swarm:
        sns.swarmplot(data=all_coefs, color='k', size=3,
                      linewidth=0, ax=ax)

    plt.axhline(y=0, color='k', linewidth=0.2)
    plt.xlim((-1, n))
    ax = plt.gca()
    ax.xaxis.set_tick_params(labelsize=x_font_size)
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


def which_x_toPrint(df, names, n_names=15):
    if df.shape[1] > 30:
        xmax = abs(df).max(axis=0)
        mthres = 0
        while (xmax > mthres).sum() > n_names:
            mthres += 0.01
            xmax = abs(df.max(axis=0))
        xpos = np.arange(df.shape[1])[(xmax > mthres)]
        xlabels = names[(xmax > mthres)]
    else:
        xpos = np.arange(df.shape[1])
        xlabels = names

    return xpos, xlabels


def _format_position(position, toPrint):
    if "'" in position.iloc[0]:
        if toPrint:
            logger.info("removing the ' character from the Chromosome Region")
        position = position.str.replace("'", "")
    if "-" in position.iloc[0]:
        if toPrint:
            logger.info("replacing the '-' with ':' character to separate " +
                        "the 'start' and 'end' in the " +
                        "Chromosome Region numbers")
        position = position.str.replace("-", ":")

    return position


def _drop_rows(onesample, dropped_rows, r2drop, reason2drop, toPrint):
    if toPrint:
        logger.info('Dropping '+str(r2drop.shape[0]) +
                    ' rows because: '+reason2drop)
    df2drop = onesample.loc[r2drop, :]
    dropped_rows = dropped_rows.append(df2drop, sort=False)
    dropped_rows.loc[r2drop, 'reason2drop'] = reason2drop
    # drop the rows
    onesample = onesample.drop(r2drop, axis=0)

    return onesample, dropped_rows


def process_oncoscan(onesample, toPrint=False, **kwargs):
    # columns with info about:
    # Chromosome Region, Event, Gene Symbols (in this order!!!)
    keep_columns = kwargs.get('keep_columns', None)
    if keep_columns is None:
        keep_columns = ["Chromosome Region", "Event", "Gene Symbols"]
        logger.warning('keep_columns kwarg is missing, ' +
                       'the following columns are assumed:\n' +
                       srt(keep_columns))
    else:
        keep_columns = keep_columns.rsplit(',')
        if len(keep_columns) > 3:
            logger.error('more than 3 column names are given!\n' +
                         'give columns with info about: ' +
                         'Chromosome Region, Event, Gene Symbols ' +
                         '(in this order!!!)')
            raise

    new_columns = kwargs.get('new_columns', None)
    if new_columns is not None:
        new_columns = new_columns.rsplit(',')
    else:
        new_columns = ['chr', 'start', 'end', 'id', 'function']

    # for each sample
    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    # choose only the columns: 'Chromosome Region', 'Event', 'Gene Symbols'
    if toPrint:
        logger.info('keep columns: '+str(keep_columns))
    onesample_small = onesample[keep_columns].copy()

    # format the Chromosome Region
    onesample_small[keep_columns[0]] = \
        _format_position(onesample_small[keep_columns[0]], toPrint)

    # change the gene symbols type (from single string to an array of strings)
    onesample_small['Gene Arrays'] = \
        onesample_small[keep_columns[2]].str.split(', ')
    # remove the row that has NaN in this column
    #  it means that the CNV does not map to a know gene
    null_genes = onesample_small['Gene Arrays'].isnull()
    if null_genes.any():
        r2drop = onesample_small.index[null_genes]
        reason2drop = 'cannot_map_gene'
        onesample_small, dropped_rows = _drop_rows(
            onesample_small, dropped_rows, r2drop, reason2drop, toPrint)
        if onesample_small.empty:
            logger.warning('after removing rows with no gene symbols, ' +
                           'there are no more CNVs for the patient')
            return onesample_small, dropped_rows

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
        list(zip(*onesample_map2genes['all'].str.split(':')))

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
        logger.info('Finished processing successfully.')

    return onesample_map2genes, dropped_rows


def filter_oncoscan(onesample, toPrint=False, **kwargs):

    col_pValue = kwargs.get('col_pValue', None)
    if col_pValue is None:
        logger.error('col_pValue kwarg is missing!')
        raise
    col_probeMedian = kwargs.get('col_probeMedian', None)
    if col_probeMedian is None:
        logger.error('col_probeMedian kwarg is missing!')
        raise
    col_probeCount = kwargs.get('col_probeCount', None)
    if col_probeCount is None:
        logger.error('col_probeCount kwarg is missing!')
        raise

    pValue_thres = kwargs.get('pValue_thres', 0.01)
    probeMedian_thres = kwargs.get('probeMedian_thres', 0.3)
    probeCount_thres = kwargs.get('probeCount_thres', 20)
    remove_missing_pValues = kwargs.get('remove_missing_pValues', False)

    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    # keep the rows we will drop
    if remove_missing_pValues:
        r2drop = onesample.index[onesample[col_pValue].isnull()]
        reason2drop = 'filter_'+col_pValue+'_missing'
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # keep the rows we will drop
    r2drop = onesample.index[onesample[col_pValue] >
                             pValue_thres]
    reason2drop = 'filter_'+col_pValue+'_'+str(pValue_thres)
    onesample, dropped_rows = _drop_rows(
        onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # keep the rows we will drop
    r2drop = onesample.index[abs(onesample[col_probeMedian]) <
                             probeMedian_thres]
    reason2drop = 'filter_'+col_probeMedian+'_' + \
        str(probeMedian_thres)
    onesample, dropped_rows = _drop_rows(
        onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # keep the rows we will drop
    r2drop = onesample.index[onesample[col_probeCount] <
                             probeCount_thres]
    reason2drop = 'filter_'+col_probeCount + \
        '_'+str(probeCount_thres)
    onesample, dropped_rows = _drop_rows(
        onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # reset index
    onesample.reset_index(drop=True, inplace=True)
    dropped_rows.reset_index(drop=True, inplace=True)

    if toPrint:
        logger.info('Finished filtering successfully.')

    return onesample, dropped_rows


def filter_excavator(onesample, toPrint=False, **kwargs):

    choose_col = kwargs.get('choose_col', 'ProbCall')
    choose_thres = kwargs.get('choose_thres', 0.95)

    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    # keep the rows we will drop
    r2drop = onesample.index[onesample[choose_col] > choose_thres]
    reason2drop = 'filter_'+choose_col+'_'+str(choose_thres)
    onesample, dropped_rows = _drop_rows(
        onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # reset index
    onesample.reset_index(drop=True, inplace=True)
    dropped_rows.reset_index(drop=True, inplace=True)

    if toPrint:
        logger.info('Finished filtering successfully.')

    return onesample, dropped_rows


def _preprocessing(patient_id, onesample, info_table,
                   withFilter, filt_kwargs,
                   withProcess, preproc_kwargs,
                   editWith, edit_kwargs,
                   toPrint):

    info_table.loc[patient_id, 'rows_in_sample'] = onesample.shape[0]
    if onesample.empty:
        logger.warning('EMPTY patient! there are no CNVs for patient ' +
                       str(patient_id))
        return onesample, info_table, pd.DataFrame([]), \
            pd.DataFrame([]), pd.DataFrame([])

    if toPrint:
        logger.info(str(onesample.shape[0]) +
                    ' rows for patient ' +
                    str(patient_id))

    if bool(filt_kwargs) and withFilter:
        if toPrint:
                logger.info('Filtering...')
        # - filter sample - #
        if editWith == 'Oncoscan':
            onesample, dropped_rows_filter_pat = \
                filter_oncoscan(onesample, toPrint=toPrint,
                                **filt_kwargs)
        elif editWith == 'ExCavator2':
            onesample, dropped_rows_filter_pat = filter_excavator(
                onesample, toPrint=toPrint, **filt_kwargs)
        else:
            logger.error('unsupported sample editor '+(editWith))
            raise

        info_table.loc[patient_id, 'rows_in_sample_filt'] = onesample.shape[0]
        if onesample.empty:
            logger.warning('after filtering ' +
                           'there are no more CNVs for patient ' +
                           str(patient_id))
            return onesample, info_table, dropped_rows_filter_pat, \
                pd.DataFrame([]), pd.DataFrame([])
        else:
            if toPrint:
                logger.info(str(onesample.shape[0]) +
                            ' rows for patient ' +
                            str(patient_id)+' after filtering')
    else:
        dropped_rows_filter_pat = pd.DataFrame([])

    if bool(preproc_kwargs) and withProcess:
        if toPrint:
                logger.info('Processing...')
        # - pre-process sample - #
        onesample, dropped_rows_process_pat = process_oncoscan(
            onesample, toPrint=toPrint, **preproc_kwargs)

        info_table.loc[
            patient_id, 'rows_in_sample_processed'] = onesample.shape[0]
        if onesample.empty:
            logger.warning('after processing ' +
                           'there are no more CNVs for patient ' +
                           str(patient_id))
            return onesample, info_table, dropped_rows_filter_pat, \
                dropped_rows_process_pat, pd.DataFrame([])
        else:
            if toPrint:
                logger.info(str(onesample.shape[0]) +
                            ' rows for patient ' +
                            str(patient_id) +
                            ' after processing')
    else:
        dropped_rows_process_pat = pd.DataFrame([])

    # np.append(onesample.columns, 'reason2drop')
    if editWith == 'Oncoscan':
        # for consistency convert to lowercase
        onesample.columns = onesample.columns.str.lower()

        # format the Chromosome Region and split pos in start and end
        if (('start' not in onesample.columns) and
                ('pos' in onesample.columns)):
            onesample['pos'] = _format_position(onesample['pos'], toPrint)
            _chr_col, onesample['start'], onesample['end'] = \
                onesample['pos'].str.rsplit(':').str

            if (('chr' not in onesample.columns) and
                    ('chrom' not in onesample.columns)):
                onesample['chr'] = _chr_col

        # verify the chromosome column name
        if 'chr' not in onesample.columns:
            if 'chrom' in onesample.columns:
                onesample['chr'] = onesample['chrom']
                onesample.drop(['chrom'], inplace=True, axis=1)
            else:
                logger.error('Columns not found: chr|chrom')
                raise

        # - edit sample - #
        onesample, dropped_rows_edit_pat = edit_oncoscan(
            onesample, patient_id, toPrint=toPrint,
            **edit_kwargs
        )

    elif editWith == 'ExCavator2':
        # - edit sample - #
        onesample, dropped_rows_edit_pat = edit_excavator(
            onesample, patient_id, toPrint=toPrint,
            **edit_kwargs
        )

    elif editWith == 'VCF':
        # - edit sample - #
        onesample, dropped_rows_edit_pat = edit_vcf(
            onesample, patient_id, toPrint=toPrint,
            **edit_kwargs
        )

    else:
        logger.error('unsupported sample editor '+(editWith))
        raise

    info_table.loc[
        patient_id, 'rows_in_sample_editted'] = onesample.shape[0]
    if toPrint:
        logger.info(str(onesample.shape[0]) +
                    ' rows for patient ' +
                    str(patient_id) +
                    ' after editting')

    return onesample, info_table, dropped_rows_filter_pat, \
        dropped_rows_process_pat, dropped_rows_edit_pat


def load_and_process_summary_file(fpaths, info_table, editWith='choose_editor',
                                  toPrint=False, **kwargs):
    load_data_csv_kwargs = kwargs.get(
        'load_data_csv_kwargs', {}
    )
    names = load_data_csv_kwargs.pop('names', None)
    if names is not None:
        if ',' in names:
            names = names.rsplit(',')
        load_data_csv_kwargs['names'] = names

    withFilter = parse_arg_type(
        kwargs.get('withFilter', False),
        bool
    )
    withProcess = parse_arg_type(
        kwargs.get('withProcess', True),
        bool
    )

    filt_kwargs = kwargs.get('filt_kwargs', {})
    preproc_kwargs = kwargs.get('preproc_kwargs', {})
    edit_kwargs = kwargs.get('edit_kwargs', {})

    # oncoscan load files from each patient
    data_or = dict()
    data = []
    info_table['rows_in_sample'] = 0
    if withFilter:
        info_table['rows_in_sample_filt'] = 0
    if withProcess:
        info_table['rows_in_sample_processed'] = 0
    info_table['rows_in_sample_editted'] = 0
    for fpath in fpaths:
        allsamples = pd.read_csv(fpath, **load_data_csv_kwargs)
        samples_colname = kwargs.get('samples_colname',
                                     allsamples.columns.values[0])

        dropped_rows_filter = pd.DataFrame()
        dropped_rows_process = pd.DataFrame()
        dropped_rows_edit = pd.DataFrame()

        for patient_id in natsorted(allsamples[samples_colname].unique()):
            if toPrint:
                logger.info('sample: '+patient_id)

            data_or[patient_id] = allsamples[allsamples[samples_colname] ==
                                             patient_id].copy()
            onesample = data_or[patient_id].copy()

            # preprocess one sample
            onesample, info_table, \
                dropped_rows_filter_pat, \
                dropped_rows_process_pat, \
                dropped_rows_edit_pat = \
                _preprocessing(
                    patient_id, onesample, info_table,
                    withFilter, filt_kwargs,
                    withProcess, preproc_kwargs,
                    editWith, edit_kwargs,
                    toPrint
                )
            if dropped_rows_filter_pat.shape[0] > 0:
                dropped_rows_filter = pd.concat(
                    [dropped_rows_filter, dropped_rows_filter_pat],
                    axis=0, sort=False)
            if dropped_rows_process_pat.shape[0] > 0:
                dropped_rows_process = pd.concat(
                    [dropped_rows_process, dropped_rows_process_pat],
                    axis=0, sort=False)
            if dropped_rows_edit_pat.shape[0] > 0:
                dropped_rows_edit = pd.concat(
                    [dropped_rows_edit, dropped_rows_edit_pat],
                    axis=0, sort=False)
            #######
            if not onesample.empty:
                data.append(onesample)
                if toPrint:
                    logger.info('finished pre-proceessing sample: ' +
                                patient_id+'\n')
            else:
                if toPrint:
                    logger.info('discarding EMPTY sample: ' +
                                patient_id+'\n')

    return data, data_or, dropped_rows_filter, \
        dropped_rows_process, dropped_rows_edit, info_table


def load_and_process_files(fpaths, info_table, editWith='choose_editor',
                           toPrint=False, **kwargs):
    load_data_csv_kwargs = kwargs.get(
        'load_data_csv_kwargs', {}
    )
    names = load_data_csv_kwargs.pop('names', None)
    if names is not None:
        if ',' in names:
            names = names.rsplit(',')
        load_data_csv_kwargs['names'] = names

    withFilter = parse_arg_type(
        kwargs.get('withFilter', False),
        bool
    )
    withProcess = parse_arg_type(
        kwargs.get('withProcess', True),
        bool
    )
    filt_kwargs = kwargs.get('filt_kwargs', {})
    preproc_kwargs = kwargs.get('preproc_kwargs', {})
    edit_kwargs = kwargs.get('edit_kwargs', {})

    fext = kwargs.get('fext', None)
    split_patID = kwargs.get('split_patID', None)

    # oncoscan load files from each patient
    data_or = dict()
    data = []
    info_table['rows_in_sample'] = 0
    if withFilter:
        info_table['rows_in_sample_filt'] = 0
    if withProcess:
        info_table['rows_in_sample_processed'] = 0
    info_table['rows_in_sample_editted'] = 0
    for fpath in fpaths:
        for filename in natsorted(os.listdir(fpath)):
            if filename.endswith(fext):
                if toPrint:
                    logger.info('filename: '+filename)

                patient_id = filename.rsplit(fext)[0]
                if split_patID is not None:
                    patient_id = patient_id.rsplit(split_patID)[0]

                if toPrint:
                    logger.info('patient_id: '+patient_id)

                sample_fpath = os.path.join(fpath, filename)
                onesample_or = pd.read_csv(
                    sample_fpath, **load_data_csv_kwargs)

                data_or[patient_id] = onesample_or.copy()

                dropped_rows_filter = pd.DataFrame()
                dropped_rows_process = pd.DataFrame()
                dropped_rows_edit = pd.DataFrame()

                onesample = data_or[patient_id].copy()

                # preprocess one sample
                onesample, info_table, \
                    dropped_rows_filter_pat, \
                    dropped_rows_process_pat, \
                    dropped_rows_edit_pat = \
                    _preprocessing(
                        patient_id, onesample, info_table,
                        withFilter, filt_kwargs,
                        withProcess, preproc_kwargs,
                        editWith, edit_kwargs,
                        toPrint
                    )

                if dropped_rows_filter_pat.shape[0] > 0:
                    dropped_rows_filter = pd.concat(
                        [dropped_rows_filter, dropped_rows_filter_pat],
                        axis=0, sort=False)
                if dropped_rows_process_pat.shape[0] > 0:
                    dropped_rows_process = pd.concat(
                        [dropped_rows_process, dropped_rows_process_pat],
                        axis=0, sort=False)
                if dropped_rows_edit_pat.shape[0] > 0:
                    dropped_rows_edit = pd.concat(
                        [dropped_rows_edit, dropped_rows_edit_pat],
                        axis=0, sort=False)
                #######
                if not onesample.empty:
                    data.append(onesample)
                    if toPrint:
                        logger.info('finished pre-proceessing sample: ' +
                                    patient_id+'\n')
                else:
                    if toPrint:
                        logger.info('discarding EMPTY sample: ' +
                                    patient_id+'\n')

    return data, data_or, dropped_rows_filter, \
        dropped_rows_process, dropped_rows_edit, info_table


def _merge_gene_values(mergeHow, functionArray, genes_startPositions,
                       genes_endPositions, toPrint):
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

    return final_dict


def _map_cnvs_to_genes(
        onesample, dropped_rows, sample_name,
        removeLOH, LOH_value, function_dict, mergeHow,
        toPrint
        ):
    # remove rows with LOH in FUNCTION !!!!!!!!!!!!!!!!!!
    if removeLOH:
        # keep the rows we will drop
        s_isLOH = (onesample['function'] == LOH_value)
        if s_isLOH.any():
            r2drop = s_isLOH.index[s_isLOH]
            reason2drop = 'LOH'
            onesample, dropped_rows = _drop_rows(
                onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # remove genes that exist in more than one chromosomes
    tmp_size = onesample.shape[0]
    # group by ID and sum over the CHROM
    # (to get all different chrom for one gene)
    chrSum = onesample.groupby(['id'])['chr'].sum()
    # save a dict with only the genes to remove
    # and an array of the diff chroms these gene exist in
    # (SLOW!!!)
    print()
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
    drop_bool = onesample['id'].isin(genes_to_remove_dict.keys())
    if drop_bool.any():
        r2drop = onesample.index[drop_bool]
        reason2drop = 'multiple_chrom'
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)
        if toPrint:
            logger.info(str(len(genes_to_remove_dict.keys())) +
                        ' unique gene IDs removed:\n' +
                        str(natsorted(genes_to_remove_dict.keys())))

    # create a new column with ID and CHR together
    onesample['CHR_ID'] = onesample['chr']+':'+onesample['id']

    # define a dict of FUNCTION values
    # onesample.FUNCTION.unique()
    if function_dict is not None:
        # create a new column with these mapped values
        onesample['FUNC_int'] = onesample['function'].map(function_dict)
    else:
        onesample['FUNC_int'] = onesample['function']
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
                        ' rows aggregated to ' +
                        str(onesample[onesample['CHR_ID'
                                                ].duplicated(keep='first')
                                      ].shape[0]) +
                        ' unique rows')
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

    drop_bool = onesample['CHR_ID'].isin(CHR_ID2drop)
    if drop_bool.any():
        r2drop = onesample.index[drop_bool]
        reason2drop = 'ampl_AND_del'
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)
        if toPrint:
            logger.info(str(len(CHR_ID2drop)) +
                        ' unique gene IDs removed:\n' +
                        str(natsorted(CHR_ID2drop)))

        # RE-group by CHR_ID and sum over the FUNCTION
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
    final_dict = _merge_gene_values(
        mergeHow, functionArray, genes_startPositions,
        genes_endPositions, toPrint)

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


def edit_oncoscan(onesample, sample_name, toPrint=True, **kwargs):

    removeLOH = parse_arg_type(
        kwargs.get('removeLOH', True),
        bool
    )
    LOH_value = kwargs.get('LOH_value', None)

    function_dict = kwargs.get('function_dict', None)
    # mergeHow: 'maxAll', 'maxOne', 'freqAll'
    mergeHow = kwargs.get('mergeHow', 'maxAll')

    # for each sample
    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    # (check_columns: chr,start,end,id,function)
    if 'value' in onesample.columns:
        check_cols = np.delete(onesample.columns.values,
                               np.where(onesample.columns.values == 'value'))
    else:
        check_cols = onesample.columns

    # remove rows with NaNs in check_columns
    df_isna = onesample[check_cols].isna()
    if toPrint:
        logger.info('Missing values for each column:')
        df_isna_sum = df_isna.sum()
        for _i in range(df_isna_sum.shape[0]):
            logger.info(str(df_isna_sum.index[_i])+'\t' +
                        str(df_isna_sum.iloc[_i]))
    if df_isna.sum().sum() > 0:
        if toPrint:
            logger.info('Remove rows with any missing values in columns:\n' +
                        str(check_cols))
        # keep the rows we will drop
        r2drop = df_isna.index[df_isna.any(axis=1)]
        reason2drop = 'missing_field'
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    df, dropped_rows = _map_cnvs_to_genes(
        onesample, dropped_rows, sample_name,
        removeLOH, LOH_value, function_dict, mergeHow,
        toPrint
        )

    return df, dropped_rows


def edit_excavator(onesample, sample_name, toPrint=True, **kwargs):
    removeLOH = parse_arg_type(
        kwargs.get('removeLOH', True),
        bool
    )
    LOH_value = parse_arg_type(
        kwargs.get('LOH_value', None),
        int
    )
    function_dict = kwargs.get('function_dict', None)
    # mergeHow: 'maxAll', 'maxOne', 'freqAll'
    mergeHow = kwargs.get('mergeHow', 'maxAll')

    # for each sample
    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    # keep_columns (here instead of process_oncoscan)
    keep_columns = kwargs.get('keep_columns', None)
    if keep_columns is None:
        keep_columns = [
            'Chromosome', 'Start', 'End', 'Call',
            'ProbCall', 'Gene_Symbol']
        logger.warning('keep_columns kwarg is missing, ' +
                       'the following columns are assumed:\n' +
                       srt(keep_columns))
    else:
        keep_columns = keep_columns.rsplit(',')
    onesample = onesample[keep_columns].copy()

    # rename those columns
    new_columns = kwargs.get('new_columns', None)
    if new_columns is not None:
        new_columns = new_columns.rsplit(',')
    else:
        new_columns = ['chr', 'start', 'end', 'function', 'probcall', 'id']
    onesample = onesample.rename(columns=dict(zip(keep_columns, new_columns)))

    # remove rows with NaNs in keep_columns
    df_isna = onesample.isna()
    if toPrint:
        logger.info('Missing values for each column:')
        df_isna_sum = df_isna.sum()
        for _i in range(df_isna_sum.shape[0]):
            logger.info(str(df_isna_sum.index[_i])+'\t' +
                        str(df_isna_sum.iloc[_i]))
    if df_isna.sum().sum() > 0:
        if toPrint:
            logger.info('Remove rows with any missing values in columns:\n' +
                        str(onesample.columns))
        # keep the rows we will drop
        r2drop = df_isna.index[df_isna.any(axis=1)]
        reason2drop = 'missing_field'
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # Remove rows with invalid Gene_Symbol - space
    c = " "
    c_name = 'Space'
    isInvalid = onesample['id'].str.contains(pat=c)
    if isInvalid.any():
        # keep the rows we will drop
        invalid_ids = onesample[isInvalid].index

        r2drop = onesample.index[isInvalid]
        reason2drop = 'invalid_gene_id_with'+c_name
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # Remove rows with invalid Gene_Symbol - colon
    c = ":"
    c_name = 'Colon'
    isInvalid = onesample['id'].str.contains(pat=c)
    if isInvalid.any():
        # keep the rows we will drop
        invalid_ids = onesample[isInvalid].index

        r2drop = onesample.index[isInvalid]
        reason2drop = 'invalid_gene_id_with'+c_name
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    df, dropped_rows = _map_cnvs_to_genes(
        onesample, dropped_rows, sample_name,
        removeLOH, LOH_value, function_dict, mergeHow,
        toPrint
        )

    return df, dropped_rows


def edit_vcf(onesample, sample_name, toPrint=True, **kwargs):
    removeLOW = parse_arg_type(
        kwargs.get('removeLOW', False),
        bool
    )
    LOW_value = parse_arg_type(
        kwargs.get('LOW_value', None),
        int
    )
    function_dict = kwargs.get('function_dict', None)
    # mergeHow: 'maxAll', 'maxOne', 'freqAll'
    mergeHow = kwargs.get('mergeHow', 'maxAll')

    # for each sample
    dropped_rows = pd.DataFrame([], columns=np.append(onesample.columns,
                                'reason2drop'))

    # separate INFO column in sub-columns
    expand_info = onesample['info'].apply(
        lambda x: pd.Series(str(x).split('|')))
    # drop rows with INFO NaN
    # keep the rows we will drop
    drop_bool = (expand_info[1].isna()) | (expand_info[3].isna())
    if drop_bool.any():
        r2drop = onesample.index[drop_bool]
        expand_info.drop(r2drop, inplace=True)
        reason2drop = 'missing_field'
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # extranct gene name info from column INFO, sub-column 3
    onesample['id'] = expand_info[3]

    # extranct mutation function info from column INFO, sub-column 1
    onesample['function'] = -1
    onesample.loc[expand_info.index[expand_info[1].str.startswith(
        'missense')], 'function'] = function_dict['missense']
    onesample.loc[expand_info.index[expand_info[1].str.startswith(
        'disruptive')], 'function'] = function_dict['nonframeshiftIndel']
    onesample.loc[expand_info.index[expand_info[1].str.startswith(
        'stop')], 'function'] = function_dict['nonsense']
    onesample.loc[expand_info.index[expand_info[1].str.startswith(
        'frameshift')], 'function'] = function_dict['frameshiftIndel']
    onesample.loc[expand_info.index[expand_info[1].str.startswith(
        'protein')], 'function'] = function_dict['frameshiftIndel']
    onesample.loc[expand_info.index[expand_info[1].str.startswith(
        'start')], 'function'] = function_dict['frameshiftIndel']
    onesample.loc[expand_info.index[expand_info[1].str.startswith(
        'splice')], 'function'] = function_dict['frameshiftIndel']

    # Remove rows with invalid Gene_Symbol - space
    c = " "
    c_name = 'Space'
    isInvalid = onesample['id'].str.contains(pat=c)
    if isInvalid.any():
        # keep the rows we will drop
        invalid_ids = onesample[isInvalid].index

        r2drop = onesample.index[isInvalid]
        reason2drop = 'invalid_gene_id_with'+c_name
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # Remove rows with invalid Gene_Symbol - colon
    c = ":"
    c_name = 'Colon'
    isInvalid = onesample['id'].str.contains(pat=c)
    if isInvalid.any():
        # keep the rows we will drop
        invalid_ids = onesample[isInvalid].index

        r2drop = onesample.index[isInvalid]
        reason2drop = 'invalid_gene_id_with'+c_name
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # drom rows with LOW and MODIFIER functions
    if removeLOW:
        # keep the rows we will drop
        s_isLOW = (onesample['function'] == LOW_value)
        if s_isLOW.any():
            r2drop = s_isLOW.index[s_isLOW]
            reason2drop = 'LOW'
            onesample, dropped_rows = _drop_rows(
                onesample, dropped_rows, r2drop, reason2drop, toPrint)

    # remove genes that exist in more than one chromosomes
    tmp_size = onesample.shape[0]
    # group by ID and sum over the CHROM
    # (to get all different chrom for one gene)
    chrSum = onesample.groupby(['id'])['chr'].sum()
    # save a dict with only the genes to remove
    # and an array of the diff chroms these gene exist in
    genes_to_remove_dict = {
        chrSum.index[idx]: np.unique(
            np.array(list(filter(None, item.rsplit('chr')))),
            return_counts=False)
        for idx, item in enumerate(chrSum)
        if len(pd.Series(
                list(filter(None, item.rsplit('chr')))
            ).value_counts().values) > 1  # 2) more than one chroms
        if len(item.rsplit('chr')) > 2  # 1) in more than one positions
    }
    # keep the rows we will drop
    # keep the rows we will drop
    drop_bool = onesample['id'].isin(genes_to_remove_dict.keys())
    if drop_bool.any():
        r2drop = onesample.index[drop_bool]
        reason2drop = 'multiple_chrom'
        onesample, dropped_rows = _drop_rows(
            onesample, dropped_rows, r2drop, reason2drop, toPrint)
        if toPrint:
            logger.info(str(len(genes_to_remove_dict.keys())) +
                        ' unique gene IDs removed:\n' +
                        str(natsorted(genes_to_remove_dict.keys())))

    # create a new column with ID and CHR together
    onesample['CHR_ID'] = onesample['chr']+':'+onesample['id']

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
                        ' rows aggregated to ' +
                        str(onesample[onesample['CHR_ID'
                                                ].duplicated(keep='first')
                                      ].shape[0]) +
                        ' unique rows')
            # these will be aggregated into one row:
            # with function with the highest frequency
            # and overall positional region

    # group by CHR_ID and sum over the FUNCTION
    # (to get all different functions for one gene)
    functionArray = onesample.groupby(['CHR_ID'])['function'].apply(
        lambda x: np.append([], x))

    # group by CHR_ID and concat all the positions
    PosArray = onesample.groupby(['CHR_ID'])['pos'].apply(
        lambda x: np.append([], x))

    # choose the min start pos
    genes_startPositions = {
        PosArray.index[idx]: int(pd.Series(item).min())
        for idx, item in enumerate(PosArray)
    }

    # choose the max end pos
    genes_endPositions = {
        PosArray.index[idx]: int(pd.Series(item).max())
        for idx, item in enumerate(PosArray)
    }

    # mergeHow: 'maxAll', 'maxOne', 'freqAll'
    final_dict = _merge_gene_values(
        mergeHow, functionArray, genes_startPositions,
        genes_endPositions, toPrint)

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


def edit_genepanel(variants, **kwargs):
    # EDIT:
    #   - map function impact to value with function_dict
    #   - substitute allele frequencies with impact values
    #   - aggregate rows to unique genes, choose how to merge
    #   - remove some patients (optional)

    function_dict = kwargs.get('function_dict', None)

    # mergeHow: 'maxAll', 'maxOne', 'freqAll'
    mergeHow = kwargs.get('mergeHow', 'maxAll')

    gene_col = kwargs.get('gene_col', None)
    func_col_txt = kwargs.get('func_col_txt', 'function')
    func_col_code = kwargs.get('func_col_code', 'functionImpact')

    cols2drop = kwargs.get('cols2drop', None)
    if cols2drop is not None:
        if ',' in cols2drop:
            cols2drop = cols2drop.rsplit(',')
    else:
        cols2drop = []

    cols2drop.extend([func_col_txt, func_col_code])

    remove_patients_list = kwargs.get('remove_patients_list', [])

    # --which genes are mutated accross all patients
    logger.info(
        str(variants[gene_col].unique().shape[0]) +
        " genes are mutated accross all patients: " +
        str(variants[gene_col].unique()))

    # --type of mutations accross all patients
    logger.info(
        "counts of " +
        str(variants[func_col_txt].unique().shape[0]) +
        " mutation types accross all patients:\n" +
        str(variants[func_col_txt].value_counts()))

    # --ONLY genes with missense mutations
    logger.info(
        str(variants[variants[func_col_txt] == 'missense'
                     ][gene_col].unique().shape[0]) +
        " genes with a missense mutation: " +
        str(variants[variants[func_col_txt] == 'missense'][gene_col].unique()))

    # --define a dict of FUNCTION values
    logger.info(
        "define a dict of FUNCTION values:\n" +
        str(function_dict))

    # MAP function impact to value with function_dict
    variants[func_col_code] = variants[func_col_txt].map(function_dict)

    # SUBSTITUTE allele frequencies with impact values
    # drop columns and keep only patients variants
    logger.info("drop the"+str(cols2drop)+"columns")
    functionImpact = variants[func_col_code].copy()
    variants.drop(cols2drop, axis=1, inplace=True)

    # map the functionImpact to the non-zero variant allele frequencies
    logger.info(
        "map the function Impact " +
        "to the non-zero variant allele frequencies")
    variants_new = variants.set_index(gene_col)
    # for each row in table
    for row in range(variants.shape[0]):
        # substitute the non-zero values with the function Impact
        variants_new.iloc[row][variants_new.iloc[row, :] > 0] = \
            int(functionImpact[row])
    variants_new = variants_new.reset_index()

    # aggregate same genes
    if mergeHow == 'maxAll':
        logger.info(
            "group by gene and keep " +
            "the max function Impact for each gene")
        variants_merged = variants_new.groupby(gene_col).max().T
    else:
        logger.error(
            "the merge option for the function Impact is not supported!\n" +
            mergeHow)

    # keep only some patients ids
    if len(remove_patients_list) > 0:
        variants_merged.drop(remove_patients_list, axis=0, inplace=True)

    logger.info(
        "\nDimensions of variants table (samples,genes):" +
        str(variants_merged.shape)+"\n")

    return variants_merged


# choose patient IDs and their order
def choose_samples(ids, dataID, choose_from=None, choose_what=None,
                   sortby=None, **sort_kwargs):
    # first order ids according to index
    ids = ids.sort_values(by=dataID, **sort_kwargs)
    # choose patients ID
    bool1 = ids[dataID].notnull()
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


# def load_clinical(fpath, **read_csv_kwargs):
#     which_dataID = read_csv_kwargs.pop('col_as_index', None)
#     patient_ids = pd.read_csv(fpath, **read_csv_kwargs)

#     # check if there are samples missing from the chosen dataID
#     # keep only the patients for which we have a sample
#     # (for CNV or var depending on our analysis)
#     # which_dataID = 'varID'
#     # which_dataID = 'cnvID'
#     if which_dataID is not None:
#         if which_dataID != patient_ids.index.name:
#             missing_samples = patient_ids[which_dataID].isna()
#             if missing_samples.any():
#                 logger.info(
#                     str(missing_samples.sum())+' missing ' +
#                     which_dataID+' samples from patient(s): ' +
#                     str(patient_ids['patient'][missing_samples].unique())
#                 )
#                 patient_ids.drop(
#                     patient_ids.index[missing_samples],
#                     axis=0, inplace=True)

#             patient_ids.set_index(which_dataID, inplace=True)

#     return patient_ids


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

    return pca


def set_heatmap_size(data):
    ds_y, ds_x = data.shape
    fs_x = 25 if ds_x > 50 else 20 if ds_x > 15 else 15 if ds_x > 10 else 10
    fs_y = 20 if ds_y > 100 else 16 if ds_y > 50 else 12 if ds_y > 25 else 8

    print_samples = True
    if ds_y > 100:
        print_samples = False

    print_genes = True
    if ds_x > 50:
        print_genes = False

    return fs_x, fs_y, print_genes, print_samples


def set_cbar_ticks(cbar, function_dict, custom_div_cmap_arg):
    if function_dict is not None:
        functionImpact_dict_r = dict(
            (v, k) for k, v in function_dict.items()
            )
        myTicks = sorted(list(functionImpact_dict_r.keys()))
        # myTicks = [0, 1, 2, 3, 4, 5]
        cbar.set_ticks(myTicks)
        cbar.set_ticklabels(pd.Series(myTicks).map(functionImpact_dict_r))
    else:
        if custom_div_cmap_arg is not None:
            cbar.set_ticks(
                np.arange(-custom_div_cmap_arg, custom_div_cmap_arg))


def edit_names_with_duplicates(df, dupl_genes_dict):
    _agg_names = df.reset_index()['gene'].values.sum()
    if ('__' in _agg_names):
        # clean the gene names if editted before
        df['cleanName'] = \
            df.reset_index()['gene']\
            .str.split('__', expand=True)[0].values

        # get the dupl genes names using the clean name
        df['dupl_genes'] = \
            df['cleanName'].map(dupl_genes_dict).values

        #  create a new name
        df['newGeneName'] = \
            df['cleanName'].values
        genes_with_dupl = set(dupl_genes_dict.keys()).intersection(
            set(df['cleanName'].values))
        df.reset_index(inplace=True, drop=False)
        df.set_index('cleanName', inplace=True)
        df.loc[genes_with_dupl, 'newGeneName'] += '__wDupl'
        df.reset_index(inplace=True, drop=False)
        df.set_index('gene', inplace=True)

    else:
        # get the dupl genes names
        df['dupl_genes'] = \
            df.reset_index()['gene']\
            .map(dupl_genes_dict).values

        #  create a new name
        genes_with_dupl = set(dupl_genes_dict.keys()).intersection(
            set(df.index.values))
        df['newGeneName'] = df.index.values
        df.loc[genes_with_dupl, 'newGeneName'] += \
            '__wDupl'

    return df
