# # from gensim.corpora import Dictionary
# # from gensim.models import ldamodel
# # # # from graphlab import numpy
# # # from numpy import *
# # import random
# # from scipy import *
# #
# # texts = [['bank','river','shore','water'],
# #         ['river','water','flow','fast','tree'],
# #         ['bank','water','fall','flow'],
# #         ['bank','bank','water','rain','river'],
# #         ['river','water','mud','tree'],
# #         ['money','transaction','bank','finance'],
# #         ['bank','borrow','money'],
# #         ['bank','finance'],
# #         ['finance','money','sell','bank'],
# #         ['borrow','sell'],
# #         ['bank','loan','sell']]
# #
# # dictionary = Dictionary(texts)
# # print dictionary
# # corpus = [dictionary.doc2bow(text) for text in texts]
# #
# # # numpy.random.seed(1) # setting random seed to get the same results each time.
# # model = ldamodel.LdaModel(corpus, id2word=dictionary, num_topics=2)
# # print  model.show_topics(2,10)
# # print "\n"
# # print "test"
# # model.get_term_topics('water')
# # model.get_term_topics('finance')
# # model.get_term_topics('bank')
# #
# # bow_water = ['bank','water','bank']
# # bow_finance = ['bank','finance','bank']
# #
# # bow = model.id2word.doc2bow(bow_water) # convert to bag of words format first
# # doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)
# #
# # print word_topics
# #
# # print phi_values
# #
# # bow = model.id2word.doc2bow(bow_finance) # convert to bag of words format first
# # doc_topics, word_topics, phi_values = model.get_document_topics(bow, per_word_topics=True)
# #
# # print word_topics
# #
# # all_topics = model.get_document_topics(corpus, per_word_topics=True)
# #
# # for doc_topics, word_topics, phi_values in all_topics:
# #     print('New Document \n')
# #     print 'Document topics:', doc_topics
# #     print 'Word topics:', word_topics
# #     print 'Phi values:', phi_values
# #     print(" ")
# #     print('-------------- \n')
# #
# # topics = model.get_document_topics(corpus, per_word_topics=True)
# # all_topics = [(doc_topics, word_topics, word_phis) for doc_topics, word_topics, word_phis in topics]
# #
# # doc_topic, word_topics, phi_values = all_topics[2]
# # print 'Document topic:', doc_topics, "\n"
# # print 'Word topic:', word_topics, "\n"
# # print 'Phi value:', phi_values
# #
# #
# #
# # for doc in all_topics:
# #     print('New Document \n')
# #     print 'Document topic:', doc[0]
# #     print 'Word topic:', doc[1]
# #     print 'Phi value:', doc[2]
# #     print(" ")
# #     print('-------------- \n')
# #
# #
# # def color_words(model, doc):
# #     import matplotlib.pyplot as plt
# #     import matplotlib.patches as patches
# #
# #     # make into bag of words
# #     doc = model.id2word.doc2bow(doc)
# #     # get word_topics
# #     doc_topics, word_topics, phi_values = model.get_document_topics(doc, per_word_topics=True)
# #
# #     # color-topic matching
# #     topic_colors = {1: 'red', 0: 'blue'}
# #
# #     # set up fig to plot
# #     fig = plt.figure()
# #     ax = fig.add_axes([0, 0, 1, 1])
# #
# #     # a sort of hack to make sure the words are well spaced out.
# #     word_pos = 1 / len(doc)
# #
# #     # use matplotlib to plot words
# #     for word, topics in word_topics:
# #         ax.text(word_pos, 0.8, model.id2word[word],
# #                 horizontalalignment='center',
# #                 verticalalignment='center',
# #                 fontsize=20, color=topic_colors[topics[0]],  # choose just the most likely topic
# #                 transform=ax.transAxes)
# #         word_pos += 0.2  # to move the word for the next iter
# #
# #     ax.set_axis_off()
# #     plt.show()
# #
# #
# # def color_words_dict(model, dictionary):
# #     import matplotlib.pyplot as plt
# #     import matplotlib.patches as patches
# #
# #     word_topics = []
# #     for word_id in dictionary:
# #         word = str(dictionary[word_id])
# #         # get_term_topics returns static topics, as mentioned before
# #         probs = model.get_term_topics(word)
# #         # we are creating word_topics which is similar to the one created by get_document_topics
# #         try:
# #             if probs[0][1] >= probs[1][1]:
# #                 word_topics.append((word_id, [0, 1]))
# #             else:
# #                 word_topics.append((word_id, [1, 0]))
# #         # this in the case only one topic is returned
# #         except IndexError:
# #             word_topics.append((word_id, [probs[0][0]]))
# #
# #     # color-topic matching
# #     topic_colors = {1: 'red', 0: 'blue'}
# #
# #     # set up fig to plot
# #     fig = plt.figure()
# #     ax = fig.add_axes([0, 0, 1, 1])
# #
# #     # a sort of hack to make sure the words are well spaced out.
# #     word_pos = 1 / len(doc)
# #
# #     # use matplotlib to plot words
# #     for word, topics in word_topics:
# #         ax.text(word_pos, 0.8, model.id2word[word],
# #                 horizontalalignment='center',
# #                 verticalalignment='center',
# #                 fontsize=20, color=topic_colors[topics[0]],  # choose just the most likely topic
# #                 transform=ax.transAxes)
# #         word_pos += 0.2  # to move the word for the next iter
# #
# #     ax.set_axis_off()
# #     plt.show()
# # if __name__ == '__main__':
# #     doc = ['bank', 'water', 'bank', 'finance', 'money', 'sell', 'river', 'fast', 'tree']
# #     color_words(model, doc)
# #
# #     color_words_dict(model, dictionary)
#
#
#     # color_words(model, bow_water)
#
#
#
#
#
#
#
#     #
#     # ## word clouds
#     # from os import path
#     # import matplotlib.pyplot as plt
#     # from wordcloud import WordCloud
#     #
#     #
#     # def terms_to_wordcounts(terms, multiplier=1000):
#     #
#     #     return " ".join([" ".join(int(multiplier * i[0]) * [i[1]]) for i in terms])
#     #
#     #
#     # wordcloud = WordCloud(font_path="new_data/Impact_Label.ttf", background_color="black").generate(terms_to_wordcounts(terms),
#     #                                                                                        1000)
#     #
#     # plt.imshow(wordcloud)
#     # plt.axis("off")
#     # plt.savefig("terms1")
#     #
#     # plt.close()
#
# import logging
# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',
# level=logging.INFO)
#
# import numpy
# import scipy
# import gensim
#
# import random
# random.seed(11091987)           #set random seed
#
#
# # load id->word mapping (the dictionary)
# id2word =  gensim.corpora.Dictionary('../newdata1/dictionary.dict')
#
# # load corpus
# ## add top line to MM file since R does not automatically add this
# ## and save new version
# with open('../dtm.mtx') as f:
#     dtm = f.read()
#     dtm = "%%MatrixMarket matrix coordinate real general\n" + dtm
#
#     with open('dtm.mtx', 'w+') as f:
#         f.write(dtm)
#
#
# corpus = gensim.corpora.MmCorpus('dtm.mtx')
#
# print id2word
# print corpus
#
# # shuffle corpus
# cp = list(corpus)
# random.shuffle(cp)
#
#  # split into 80% training and 20% test sets
# p = int(len(cp) * .8)
# cp_train = cp[0:p]
# cp_test = cp[p:]
#
# import time
# start_time = time.time()
#
# lda = gensim.models.ldamodel.LdaModel(corpus=cp_train, id2word=id2word,num_topics=25,update_every=1, chunksize=1000, passes=2)
#
# elapsed = time.time() - start_time
# print('Elapsed time: '),
# print elapsed
#
#
# print lda.show_topics(topics=-1, topn=10, formatted=True)
#
# print('Perplexity: '),
# perplex = lda.bound(cp_test)
# print perplex
#
# print('Per-word Perplexity: '),
# print numpy.exp2(-perplex / sum(cnt for document in cp_test for _, cnt in  document))
#
# elapsed = time.time() - start_time
# print('Elapsed time: '),
# print elapsed
# ignore_namespaces = 'did above s.n r tm i  / || xv'.split()
# print ignore_namespaces

# print lda_model.show_topics(10,20)
#     # lmlist, c_v = evaluate_graph( model= lda_model,dictionary=dictionary, corpus=mm_corpus, limit=10)

    # select top 50 words for each of the 20 LDA topics
    # top_words = [[word for _, word in lda_model.show_topic(topicno, topn=50)] for topicno in
    #              range(lda_model.num_topics)]
    # print(top_words)
    #
    # # get all top 50 words in all 20 topics, as one large set
    # all_words = set(itertools.chain.from_iterable(top_words))
    #
    # print("Can you spot the misplaced word in each topic?")
    #
    # # for each topic, replace a word at a different index, to make it more interesting
    # replace_index = np.random.randint(0, 10, lda_model.num_topics)
    #
    # replacements = []
    # for topicno, words in enumerate(top_words):
    #     other_words = all_words.difference(words)
    #     replacement = np.random.choice(list(other_words))
    #     replacements.append((words[replace_index[topicno]], replacement))
    #     words[replace_index[topicno]] = replacement
    #     print("%i: %s" % (topicno, ' '.join(words[:10])))
    # # # mm_corpus=[dictionary.doc2bow(doc) for doc in doc_stream]
    # cm = CoherenceModel(model=lda_model, corpus=mm_corpus, dictionary=dictionary, coherence='u_mass')
    # print cm.get_coherence()


    # print('Number of unique tokens: %d' % len(dictionary))
    # print('Number of documents: %d' % len(mm_corpus))
    # # for i in range(1,10):
    # goodLdaModel = LdaModel(corpus=mm_corpus, id2word=dictionary, iterations=50, num_topics=10)
    # badLdaModel = LdaModel(corpus=mm_corpus, id2word=dictionary, iterations=15, num_topics=10)

    # goodcm = CoherenceModel(model=goodLdaModel, corpus=mm_corpus, dictionary=dictionary, coherence='u_mass')
    # badcm = CoherenceModel(model=badLdaModel, corpus=mm_corpus, dictionary=dictionary, coherence='u_mass')
    # print goodcm.get_coherence()
    # print badcm.get_coherence()
        # print "---------------------"

    # goodcv = CoherenceModel(model=goodLdaModel, texts=doc_stream, dictionary=dictionary, coherence='c_v')
    # badcv = CoherenceModel(model=badLdaModel, texts=doc_stream, dictionary=dictionary, coherence='c_v')

    # print goodcv.get_coherence()
    # print badcv.get_coherence()


# try:
#     model1 = LdaVowpalWabbit('/home/sparkle/PycharmProjects/LDATopicModelingFinal/new_data', corpus=mm_corpus, num_topics=2, id2word=dictionary, passes=50)
#     model2 = LdaVowpalWabbit('/home/sparkle/PycharmProjects/LDATopicModelingFinal/new_data', corpus=mm_corpus, num_topics=2, id2word=dictionary, passes=1)
#     cm1 = CoherenceModel(model=model1, corpus=mm_corpus, coherence='u_mass')
#     cm2 = CoherenceModel(model=model2, corpus=mm_corpus, coherence='u_mass')
#     print cm1.get_coherence()
#     print cm2.get_coherence()
# except:
#     print "eee"



# lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=mm_corpus, texts=stream.word, limit=10)
#     print("mm corpus")
# #
#     tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=dictionary)
#     print(tfidf_model.__sizeof__())
# #
# #
#     # lda_model = gensim.models.LdaModel(tfidf_model[mm_corpus], num_topics=10, id2word=dictionary, passes=20, alpha='auto')
# #     # lda_model = gensim.models.LdaModel(tfidf_model[mm_corpus], num_topics=10, id2word=dictionary)
# #     lda_model.save('../new_data/ldamodel.lda')
# #     topic =  lda_model.show_topics(4,10)
#     lda_model = gensim.models.LdaModel.load('../new_data/ldamodel.lda')

    # top_topics = lda_model.top_topics(mm_corpus, num_words=10)
    # print top_topics

    # Average topic coherence is the sum of topic coherences of all topics, divided by the number of topics.
    # avg_topic_coherence = sum([t[1] for t in top_topics]) / 10
    # print('Average topic coherence: %.4f.' % avg_topic_coherence)

    # from pprint import pprint
    # pprint(top_topics)
#     topics = lda_model.show_topics(num_topics=8, num_words=200)



#get terms in the selected topic
# topic_terms = lda_model.get_topic_terms(3, topn=30)
# #get the words according to the trem numbers and create a list of lists
# w_cloud = {}
# w_cloud['words'] = [dictionary[e[0]] for e in topic_terms]
# w_cloud['freq'] = [e[1] for e in topic_terms]
# tok = pd.DataFrame(w_cloud).values.tolist()
# tok = [[e[1], e[0]] for e in tok]
#
# wordcloud = WordCloud().generate_from_frequencies(tok)
# plt.figure(figsize=(10,7))
# plt.imshow(wordcloud)
# plt.axis("off")
# plt.show()

from gensim.parsing.preprocessing import STOPWORDS

print STOPWORDS