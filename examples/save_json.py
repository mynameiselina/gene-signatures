import os
import json

script_path = os.path.dirname(__file__)
# script_fname = os.path.basename(__file__).rsplit('.')[0]
print(script_path)


config_kwargs = {
    # script set up params
    'saveReport': True,
    'txt_label': 'set up Oncoscan data',
    'toPrint': True,
    # 'reportName': script_fname,
    'load_files': False,
    'editWith': 'Oncoscan',
    'withFilter': False,
    'withPreprocess': True,
    'filt_kwargs': {
        'col_pValue': 'Call PValue',
        'col_probeMedian': 'Probe Median',
        'col_probeCount': 'Probes',
        'pValue_thres': 0.01,
        'probeMedian_thres': 0.3,
        'probeCount_thres': 20,
        'remove_missing_pValues': False
    },
    'chr_col': 'chr_int',
    'gene_id_col': 'gene',
    'remove_patients': None,
    'sample_info_fname': '20180704_emca.csv',
    'sample_info_table_index_colname': 'Oncoscan_ID',
    'sample_info_table_sortLabels': 'TP53_mut5,FOXA1_mut5',
    'plot_kwargs': {
        'cmap_custom': None,
        'vmin': -4,
        'vmax': 4,
        'highRes': True
    },

    # params only for internal usage in function
    # load_and_process_summary_file() in core.py
    'samples_colname': 'Sample',
    'preproc_kwargs': {
        # columns with info about:
        # Chromosome Region, Event, Gene Symbols
        # (in this order!!!)
        'keep_columns': ['Chromosome Region', 'Event', 'Gene Symbols'],
    },
    'comment': "#",
    'removeLOH': True,
    'function_dict': {
        'Homozygous Copy Loss': -4,
        'CN Loss': -2,
        'CN Gain': 2,
        'High Copy Gain': 4,
        'LOH': -1
    },
    'mergeHow': 'maxAll',

    # set directories
    'input_directory': 'input,endometrial',
    'oncoscan_directory': 'fromNexusExpress',
    'oncoscan_files': 'DS_097_all_samples.txt,18044_all_samples.txt',
    'output_directory': 'output',
    'DEBUG': True
}

fpath = os.path.join(script_path,'config.json')
with open(fpath, 'w') as fp:
    json.dump(config_kwargs, fp, indent=4)