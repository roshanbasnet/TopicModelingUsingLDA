# import gensim
# from gensim.models import CoherenceModel
# from gensim.models import LdaModel
import matplotlib.pyplot as plt
#
#
# def evaluate_graph(dictionary, corpus, texts, limit):
#     """
#     Function to display num_topics - LDA graph using c_v coherence
#
#     Parameters:
#     ----------
#     dictionary : Gensim dictionary
#     corpus : Gensim corpus
#     limit : topic limit
#
#     Returns:
#     -------
#     lm_list : List of LDA topic models
#     c_v : Coherence values corresponding to the LDA model with respective number of topics
#     """
#     c_v = []
#     lm_list = []
#     for num_topics in range(1, limit):
#         lm = LdaModel(corpus=corpus, num_topics=num_topics, id2word=dictionary)
#         lm_list.append(lm)
#         cm = CoherenceModel(model=lm, texts=texts, dictionary=dictionary, coherence='c_v')
#         c_v.append(cm.get_coherence())
#
#     # Show graph
#     x = range(1, limit)
#     plt.plot(x, c_v)
#     plt.xlabel("num_topics")
#     plt.ylabel("Coherence score")
#     plt.legend(("c_v"), loc='best')
#     plt.show()
#
#     return lm_list, c_v
#
# if __name__ == '__main__':
#     MM_new_corpus = 'new_data/newcorpus.mm'
#
#     dictionary = gensim.corpora.Dictionary.load('data/dictionary.dict')
#     # # # Load the corpus created in the previous cells...no need to repeat the previoussteps.
#     mm_corpus = gensim.corpora.MmCorpus(MM_new_corpus)
#     print("mm corpus")
#
#     tfidf_model = gensim.models.TfidfModel(mm_corpus, id2word=dictionary)
#     lmlist, c_v = evaluate_graph(dictionary=dictionary, corpus=mm_corpus, texts=dictionary, limit=10)

# var = df.groupby('BMI').Sales.sum()
#
# fig = plt.figure()
# ax1 = fig.add_subplot(1,1,1)
# ax1.set_xlabel('BMI')
# ax1.set_ylabel('Sum of Sales')
# ax1.set_title("BMI wise Sum of Sales")
# var.plot(kind='line')
import itertools

import gensim
# from theano.gradient import np
# import n
import numpy as np
from ldamodel import iter_newsgroup, TRAIN_OUT_DATA_FILE, TEST_OUT_FILE

doc_stream = (tokens for _, tokens in iter_newsgroup(TEST_OUT_FILE))

# doc_stream = (tokens for _, tokens in _newiiter('./data/simplewiki-20140623-pages-articles.xml.bz2'))  # generator
test_docs = list(itertools.islice(doc_stream, 300, 400))


id2word = gensim.corpora.Dictionary(doc_stream)



def intra_inter(model, test_docs, num_pairs=100):
    # split each test document into two halves and compute topics for each half
    part1 = [model[id2word.doc2bow(tokens[: len(tokens) / 2])] for tokens in test_docs]
    part2 = [model[id2word.doc2bow(tokens[len(tokens) / 2:])] for tokens in test_docs]

    # print computed similarities (uses cossim)
    print("average cosine similarity between corresponding parts (higher is better):")
    print(np.mean([gensim.matutils.cossim(p1, p2) for p1, p2 in zip(part1, part2)]))

    random_pairs = np.random.randint(0, len(test_docs), size=(num_pairs, 2))
    print("average cosine similarity between 10,000 random parts (lower is better):")
    print(np.mean([gensim.matutils.cossim(part1[i[0]], part2[i[1]]) for i in random_pairs]))


#
#
#     # num topics
#     parameter_list = [10, 20, 30, 45, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150]
#
#     # alpha / eta
#     # parameter_list=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 1.5]
#
#     # we can sample if we like
#     print len(mm_corpus)
#     cp = random.sample(mm_corpus, 483)
#
#     # shuffle corpus
#     # cp = list(corpus)
#     # random.shuffle(cp)
#
#     # split into 80% training and 20% test sets
#     p = int(len(cp) * .5)
#     cp_train = cp[0:p]
#     cp_test = cp[p:]
#
#     # for num_topics_value in num_topics_list:
#
#     grid = []
#     for parameter_value in parameter_list:
#         # print "starting pass for num_topic = %d" % num_topics_value
#         print "starting pass for parameter_value = %.3f" % parameter_value
#         start_time = time.time()
#
#         # run model
#         model = gensim.models.LdaModel(corpus=cp_train, id2word=dictionary, num_topics=parameter_value,
#                                          chunksize=3125,
#                                          passes=25, update_every=0, alpha=None, eta=None, decay=0.5)
#
#         # show elapsed time for model
#         elapsed = time.time() - start_time
#         print "Elapsed time: %s" % elapsed
#
#         perplex = model.bound(cp_test)
#         print "Perplexity: %s" % perplex
#         try:
#             grid[parameter_value].append(perplex)
#         except:
#             continue
#
#         # per_word_perplex = np.exp2(-perplex / sum(cnt for document in cp_test for _, cnt in document))
#         # print "Per-word Perplexity: %s" % per_word_perplex
#         # grid[parameter_value].append(per_word_perplex)
#
#
#
#
