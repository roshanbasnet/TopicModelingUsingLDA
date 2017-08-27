# import json
# import gensim
# import itertools
#
# from gensim.parsing import STOPWORDS
# from gensim.utils import simple_preprocess
#
# from ldamodel import iter_newsgroup, TRAIN_OUT_DATA_FILE, extract
#
# def process_message(message):
#     message = gensim.utils.to_unicode(message, 'latin1').strip()
#     blocks = message.split(u'\n\n')
#     # skip email headers (first block) and footer (last block)
#     content = u'\n\n'.join(blocks[1:])
#     return content
#
# def iter_20newsgroups(fname, log_every=None):
#
#     # content = ''
#     for line in open(fname):
#         title, text = extract(json.loads(line))
#         # print text
#         message = gensim.utils.to_unicode(text, 'latin1').strip()
#         blocks = message.split(u'\n\n')
#         content = u'\n\n'.join(blocks[1:])
#     yield content
#
# def tokenize(text):
#     return [token for token in simple_preprocess(text) if token not in STOPWORDS]
#
#
#
#
# if __name__ == '__main__':
#
#         # text = process_message(text)
#         for it in iter_20newsgroups(TRAIN_OUT_DATA_FILE):
#             print it
from nltk.corpus import stopwords

tokens = 'This is the best Topc modeling Using LDA'.split()
# tokens = ['e.g','i.e','etc','did','does']
stopWords = set(stopwords.words('english'))

stopped_tokens = set([i for i in tokens if i not in stopWords])
stopped_tokens = []
# for i in  tokens:
#     if i not in stopwords:
#         stopped_tokens= i
ignore_namespaces = ['e.g','This','i.e','etc','did','does','anyone', 'above','hi','+44','x11', 's.n', 'r', 't','c','k','w','g','***','b','nt','tm', 'i', '/', '||', 'xv','x','y','z','ca','b''i','s','e','thanks','in','advance']

# for i in stopped_tokens:
#     if i not in ignore_namespaces and len(i)>2:
#         stopped_tokens = i
stopped_tokens = [i for i in stopped_tokens if i not in ignore_namespaces and len(i) >2]

print stopped_tokens

