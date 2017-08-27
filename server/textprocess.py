import re
import string
import nltk
import gensim
import itertools
import networkx
from gensim.parsing import STOPWORDS

TAGGER = nltk.tag.perceptron.PerceptronTagger()
NP_CHUNKER = nltk.RegexpParser(
    '''
    NP:    {<DT><WP><VBP>*<RB>*<VBN><IN><NN>}
           {<NN|NNS|NNP|NNPS><IN>*<NN|NNS|NNP|NNPS>+}
           {<JJ>*<NN|NNS|NNP|NNPS><CC>*<NN|NNS|NNP|NNPS>+}
           {<JJ>*<NN|NNS|NNP|NNPS>+}
    '''
)
CUSTOM_STOPS = set(['.net', '.com', '.edu', '.org'])
ENGLISH_STOPS = set(nltk.corpus.stopwords.words('english'))
ALL_STOPS = ENGLISH_STOPS.union(CUSTOM_STOPS)
PUNCT = set(string.punctuation)
NP_TAGSET = set(['NNP', 'NNPS'])


def text_to_token(document):
    ''' simple split of document into word tokens '''
    return [token for token in nltk.word_tokenize(document)]

def text_to_tagged_sentences(document):
    ''' convert document to list of sentences
        each sentence as list of part-of-speech tagged words

    '''
    return [
        nltk.tag._pos_tag(token, None, TAGGER)
        # _before, _blankline, after = text.partition('\n\n')
        for token in [nltk.word_tokenize(sent)
                      for sent in nltk.sent_tokenize(document)]
    ]

def tagged_sentences_to_npp_unigrams(tags):
    ''' filter tagged sentences to list of proper nouns
    '''
    return [word.lower()
            for token in tags
            for word, pos in token
            if (pos in NP_TAGSET) and (word not in CUSTOM_STOPS)]

def tagged_sentences_to_bagofnp(tags):
    ''' tagged sentences to flat list of noun phrases
    '''
    bow = []
    bow_append = bow.append
    for tag in tags:
        tree = NP_CHUNKER.parse(tag)
        for subtree in tree.subtrees():
            if subtree.label() == 'NP':
                bow_append(' '.join(word.lower() for word, tag in subtree.leaves()
                                    if word not in PUNCT and not any(w in word for w in CUSTOM_STOPS)))
    return bow

def tagged_sentences_to_word_candidates(tags,
                                        good_tags=set(['JJ','JJR','JJS','NN','NNP','NNS','NNPS'])):
    ''' filter and transform tagged senetences to flat list of word candidats for text-rank
    '''
    return [word.lower() for word, tag in itertools.chain.from_iterable(tags)
            if tag in good_tags and word.lower() not in ENGLISH_STOPS
and not all(char in PUNCT for char in word)]


def split_words(text, stopwords):
    """
    Break text into a list of single words. Ignore any token that falls into
    the `stopwords` set.

    """
    for word in  text:
        if word not in stopwords and len(word) > 3:
            return word

