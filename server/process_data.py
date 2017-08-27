import os
import gensim


def process_topic(topic):
    """
    0.039*engineer + 0.033*medical + 0.032*life + 0.032*museum + 0.027*role + 0.025*service + 0.024*management + 0.023*born + 0.023*held + 0.022*location
    """
    return sorted(
        [t.split('*') for t in topic.split('+')],
        key=lambda x: x[0],
        reverse=True)

def get_topics(lda_path):
    lda_model = gensim.models.LdaModel.load('new_data/ldamodel.lda')
    # lda = gensim.models.LdaModel.load(lda_path)
    print lda_model
    
    f = open('bbc_topics.json', 'w')
    f.write('{\n')
    for topic_no in xrange(lda_model.num_topics):
        word_weight = lda_model.show_topic(topic_no)
        total_weight = sum([float(s[0]) for s in word_weight])
        content = [[w, round(float(wt)/total_weight * 100, 2) if total_weight else 0] for wt, w in word_weight]
        print content
        f.write("topic" + str(topic_no) + ": " + str(content) + ",\n")
    f.write('}')
    f.close()

def get_wiki_files_list(docs_path):
    f = open('static/json/wiki1_files.json', 'w')
    f.write('{\n')
    for fl in os.listdir(docs_path):
        print fl
        if fl[0] == '.':
            continue
        with open(os.path.join(docs_path, fl), 'r') as tf:
            f.write(tf.readlines()[0].replace("\n", "").translate(None, "'!@#$?.;%:&,") + ': "' + fl + '",\n')
    f.write('}')
    f.close()


if __name__ == '__main__':
   # print process_topic('0.039*engineer + 0.033*medical + 0.032*life + 0.032*museum + 0.027*role + 0.025*service + 0.024*management + 0.023*born + 0.023*held + 0.022*location')
   #  get_topics('/home/sparkle/PycharmProjects/topics-visualization-master/new_data/ldamodel.lda')
   print get_wiki_files_list('/home/sparkle/PycharmProjects/topics-visualization-master/static/json/')

