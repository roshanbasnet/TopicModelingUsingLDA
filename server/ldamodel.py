import codecs
import io
import os
import json
import os
import json
import pickle
import random
import re
import logging,bz2
import itertools
from sympy.plotting.intervalmath.tests.test_interval_functions import np
# from tokenize import tokenize

import gensim
import time

import nltk
import unicodedata
from gensim.models import CoherenceModel
from gensim.models import LdaModel
from nltk.corpus import stopwords

import pyLDAvis.gensim
import pyLDAvis.gensim
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud, wordcloud

from process_data import process_topic
from  Ldatest import tokenize
from textprocess import text_to_tagged_sentences, tagged_sentences_to_bagofnp, tagged_sentences_to_word_candidates, \
    split_words
import matplotlib.pyplot as plt


TRAIN_DATA = '../new_data'
TRAIN_OUT_DATA_FILE = '../new_data/merg_data.json'

# file for text analysis
MM_new_corpus = '../new_data/newcorpus.mm'

# files for text analysis
MM_BOW_CORPUS = '../new_data/20newsgroup_bow.mm'
# MM_CORPUS =  '../data/newsgroup_bow.mm'
# G_JSON_OUTPUT = '../data/force.json'


def merge_tojsonl(srcdir, outfilename):
    ''' merge all source files into one jsonl for nlp streaming
    '''
    rgx_newline = re.compile('\n')  # sub with ' '
    bad_chars = bad_chars = '<>^'

    rgx_general = re.compile('[%s]' % bad_chars)  # sub with ''
    # print rgx_general

    def filter_filename(fname):
        ''' util function to filter out unwanted files by filename '''
        return fname[0] == '.'

    def pre_process_doc(content):
        ''' util function to sub for any troublesome chars '''
        return rgx_general.sub('', rgx_newline.sub(' ', content))

    def file_path_iter(source_dir):
        ''' iterator over dir that generates full paths to files of interest '''
        return (os.path.join(root, f)
                for root, dirs, files in os.walk(source_dir)
                for f in files if not filter_filename(f)
                )

    def write_to_file(path, dst):
        ''' write dictionary of source document info to dst '''

        with codecs.open(path, "r",encoding='utf-8', errors='ignore') as src:
        # with open(path, 'r') as src:
            splitp = path.encode('utf-8').split('/')

            json.dump({'docid': splitp[-1],
                       'label': splitp[-2],
                       'doc': pre_process_doc(src.read())
                       }, dst)
            dst.write('\n')

    ''' actually do stuff '''
    with open(outfilename, 'w') as dst:
        for path in file_path_iter(srcdir):
            write_to_file(path, dst)




def head(stream, n=10):
    ''' convenience fnc: return the first `n` elements of the stream, as plain list '''
    return list(itertools.islice(stream, n))

def extract(jsonobj):
    return jsonobj['docid'], jsonobj['doc']



def iter_newsgroup(merged_file):
    ''' yield tokens from each article in merged_file as a `(title, tokens)` 2-tuple. '''
    ignore_namespaces = ['ax','e.g','i.e','etc','did','does','anyone', 'above','hi','+44','x11', 's.n', 'r', 't','c','k','w','g','***','b','nt','tm', 'i', '/', '||', 'xv','x','y','z','ca','b''i','s','e','thanks','in','advance']
    for line in open(merged_file):
        title, text = extract(json.loads(line))

        # sents = filter_wiki(text)
        tokens = ''

        sents = text_to_tagged_sentences(text)
        # gensim.utils.tokenize(text, lower=True)
        # tokens = tokenize(sents)
        tokens = tokenize(text)
        # tokens = gensim.utils.lemmatize(tokens)
        tokens = tagged_sentences_to_bagofnp(sents)
        stopWords = set(stopwords.words('english'))
        stopped_tokens = [i for i in tokens if i not in stopWords]
        stopped_tokens = [i for i in stopped_tokens if i not in ignore_namespaces and len(i) >3]
        # tokens = split_words(tokens,stopWords)

        # if len(stopped_tokens) < 50 or any(title.startswith(ns + ':') for ns in ignore_namespaces):
        #
        #     continue  # ignore short articles and various meta-articles
        # for tok in tokens
        yield title, stopped_tokens


def evaluate_graph(dictionary, corpus, texts, limit):
    """
    Function to display num_topics - LDA graph using c_v coherence

    Parameters:
    ----------
    dictionary : Gensim dictionary
    corpus : Gensim corpus
    limit : topic limit

    Returns:
    -------
    lm_list : List of LDA topic models
    c_v : Coherence values corresponding to the LDA model with respective number of topics
    """
    c_v = []
    lm_list = []
    for num_topics in range(1, limit):
        lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
        lm_list.append(lm)
        cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
        c_v.append(cm.get_coherence())

    # Show graph
    x = range(1, limit)
    plt.plot(x, c_v)
    plt.xlabel("num_topics")
    plt.ylabel("Coherence score")
    plt.legend(("c_v"), loc='best')
    plt.show()

    return lm_list, c_v

class NewsGroupCorpus(object):
    def __init__(self, merge_file, dictionary, clip_docs=None):
        '''
        Parse the first `clip_docs` Wikipedia documents from file `dump_file`.
        Yield each document in turn, as a list of tokens (unicode strings).

        '''
        self.merge_file = merge_file
        self.dictionary = dictionary
        self.clip_docs = clip_docs

    def __iter__(self):
        self.titles = []
        for title, tokens in itertools.islice(iter_newsgroup(self.merge_file), self.clip_docs):
            self.titles.append(title)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        return self.clip_docs

def write_to_json(file,model):
    f = open(file, 'w')

    #f.write('{\n')
    conDic2 = {}
    count = 0
    for topic_no in xrange(model.num_topics):
        word_weight = model.show_topic(topic_no)
        total_weight = sum([float(s[1]) for s in word_weight])
        #print total_weight
        content = [[ unicodedata.normalize('NFKD', wt).encode('ascii', 'ignore'), round(float(w) / total_weight * 100, 2) if total_weight else 0] for wt, w in word_weight]
        conDic = []
        for con in  content:
            conDic.append([con[0], con[1]])

        count = count + 1
        print count
        conDic2['topic' + `count`] = conDic

    # print json.dumps(conDic2)
    f.write(json.dumps(conDic2,sort_keys=True,indent=4,separators=(',', ': '))+'\n')
    f.close()

def sample_file(file):
    f = open(file,'r')
    sample = f.read()
    return sample







# if __name__ == '__main__':
#     merge_tojsonl(TRAIN_DATA, TRAIN_OUT_DATA_FILE)
#     doc_stream = (tokens for _, tokens in iter_newsgroup(TRAIN_OUT_DATA_FILE))
#     # tfidf_vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=1000, stop_words='english')
#     # doc_stream = tfidf_vectorizer.fit_transform(doc_stream)
# #     print("doc_stream")
# #
#     id2word = gensim.corpora.Dictionary(doc_stream)
#     id2word.filter_extremes(no_below=10, no_above=0.10)
# #     id2word.save('../new_data/dictionary.dict')
# #
# #     print "id2word"
# #     # load the dictionary
#     dictionary = gensim.corpora.Dictionary.load('../new_data/dictionary.dict')
# #
# #     # create a stream of bag-of-words vectors
#     newsgroup_corpus = NewsGroupCorpus(TRAIN_OUT_DATA_FILE, dictionary)
# #     #
# #     # # store all those bag-of-words vectors into a file
# #     gensim.corpora.MmCorpus.serialize(MM_new_corpus, newsgroup_corpus)
# #     #
# #     print "mmcorpus"
# #     #
# #     # Load the corpus created in the previous cells...no need to repeat the previoussteps.
#     mm_corpus = gensim.corpora.MmCorpus(MM_new_corpus)
# #     print 'tessst'
#     lda_model = gensim.models.LdaModel(mm_corpus, num_topics=2, id2word=dictionary, passes=20,alpha='auto')
# #     # lda_model = gensim.models.LdaModel.load('../new_data/ldamodel.lda')
#     print lda_model.show_topics(10,20)
# #
#     # vis = pyLDAvis.gensim.prepare(lda_model, mm_corpus, dictionary)
#
#     print "test"
#
#     # vis = pyLDAvis.gensim.prepare(lda_model, mm_corpus, dictionary)
#     # pyLDAvis.save_html(vis, '/home/sparkle/PycharmProjects/LDATopicModelingFinal/templates/lda.html')
#     print lda_model.num_topics
#     write_to_json('../static/json/comp_graphics_topics.json',lda_model)
#
