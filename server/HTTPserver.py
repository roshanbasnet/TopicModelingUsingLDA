import os

import itertools

import gensim
import pyLDAvis
import unicodedata
from flask import Flask, jsonify
from flask import render_template
from flask import request
# from flask import send_from_directory
from flask.ext.cors import CORS
import ldamodel as lda
import json

tmp_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')

app = Flask(__name__, template_folder='../templates', static_folder='../static')
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
# app = Flask(__name__)

@app.route('/')
def index():
    return render_template(
        'home.html',
        data=[
            {'name':'/home/sparkle/PycharmProjects/LDATopicModelingFinal/new_data'},
            {'name':'/home/sparkle/PycharmProjects/LDATopicModelingFinal/newdata1'},
            {'name':'/home/sparkle/PycharmProjects/LDATopicModelingFinal/data/20news-bydate-train'},
            {'name':'/home/sparkle/PycharmProjects/LDATopicModelingFinal/data/20news-bydate-test'}
        ])


@app.route("/bbc")
def bbc():
    return render_template('bbc_index.html')
@app.route('/wiki')
def wiki():
    return render_template('index.html')


@app.route("/news")
def jslda():
    return render_template('ldanew.html')

@app.route('/clod')
def wcloud():
    return render_template('jslda.html')

@app.route('/20news')
def twenty_news():
    return render_template('20news_topics.html')

@app.route('/sample', methods=['GET','POST'])
def show_sample():
    file = '../new_data/comp.graphics/38758'
    read_file = lda.sample_file(file)
    # print read_file
    testfile = jsonify(result = read_file)
    print(testfile)
    return jsonify(result=read_file)
    # print "I am in sample urel"

    # return render_template('home.html',read_file=read_file)

@app.route("/upload" , methods=['POST'])

def upload():
    TRAIN_DATA = request.form.get('comp_select')
    print TRAIN_DATA
    topic = int(request.form.get('topics'))
    TRAIN_OUT_FILE = '../newdata1/merged_train.jsonl'
    MM_CORPUS = '../newdata1/20newsgroup_bow.mm'

    # lda.merge_tojsonl(TRAIN_DATA, TRAIN_OUT_FILE)
    print topic
    #
    doc_stream = (tokens for _, tokens in lda.iter_newsgroup(TRAIN_OUT_FILE))
    # # for i in doc_stream:
    # #     print i
    print("doc_stream")
    #
    id2word = gensim.corpora.Dictionary(doc_stream)
    id2word.filter_extremes(no_below=10, no_above=0.10)
    id2word.save('../newdata1/dictionary.dict')

    #load the dictionary
    dictionary = gensim.corpora.Dictionary.load('../newdata1/dictionary.dict')

    #create a stream of bag-of-words vectors
    newsgroup_corpus = lda.NewsGroupCorpus(TRAIN_OUT_FILE, dictionary)
    print "dict"
    # store all those bag-of-words vectors into a file
    gensim.corpora.MmCorpus.serialize(MM_CORPUS, newsgroup_corpus)

    mm_corpus = gensim.corpora.MmCorpus(MM_CORPUS)
    print dictionary
    print("mm corpus")

    tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=dictionary)
    print(tfidf_model.__sizeof__())
    #
    print "lda"
    lda_model = gensim.models.LdaModel(tfidf_model[mm_corpus], num_topics=topic, id2word=dictionary, passes=20, alpha='auto')
    lda_model.save('../newdata1/ldamodel.lda')
    print topic
    lda_model = gensim.models.LdaModel.load('../newdata1/ldamodel.lda')
    print "tseee"

    lda.write_to_json('../static/json/comp_graphics_topics.json', lda_model)

    # vis = pyLDAvis.gensim.prepare(lda_model, mm_corpus, dictionary)
    # print "test"
    # vis = pyLDAvis.gensim.prepare(lda_model, mm_corpus, dictionary)
    # pyLDAvis.save_html(vis, '/home/sparkle/PycharmProjects/LDATopicModelingFinal/templates/ldanew.html')
    # print "test3"




    return render_template('20news_topics.html')


if __name__ == '__main__':
    app.run(debug=True)
    app.config["CACHE_TYPE"] = "null"
