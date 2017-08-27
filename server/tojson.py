# import os
# import json
# import re
#
# from awscli.customizations.s3.fileinfo import write_to_file
#
#
# def merge_tojsonl(srcdir, outfilename):
#
#     ''' merge all source files into one jsonl for nlp streaming
#     '''
#
#     rgx_newline = re.compile('\n') # sub with ' '
#     bad_chars = bad_chars = '<>^'
#     rgx_general = re.compile('[%s]' % bad_chars) # sub with ''
#
#
#     def filter_filename(fname):
#         ''' util function to filter out unwanted files by filename '''
#         return fname[0] == '.'
#
#
#     def pre_process_doc(content):
#         ''' util function to sub for any troublesome chars '''
#         return rgx_general.sub('', rgx_newline.sub(' ', content))
#
#
#     def file_path_iter(source_dir):
#         ''' iterator over dir that generates full paths to files of interest '''
#         return (os.path.join(root, f)
#                 for root, dirs, files in os.walk(source_dir)
#                 for f in files if not filter_filename(f)
#                )
#
#
#     def write_to_file(path, dst):
#         ''' write dictionary of source document info to dst '''
#         with open(path, 'r', encoding='utf-8', errors='ignore') as src:
#             splitp = path.split('/')
#             json.dump({'docid': splitp[-1],
#                        'topic': splitp[-2],
#                        'doc': pre_process_doc(src.read())
#                        }, dst)
#             dst.write('\n')
#
#
#     ''' actually do stuff '''
#     with open(outfilename, 'w') as dst:
#         for path in file_path_iter(srcdir):
# write_to_file(path, dst)

import json
from pprint import pprint

with open('data/merged_test.jsonl') as data_file:
    data = json.load(data_file)

pprint(data)