import numpy as np
import lda
import lda.datasets
from utilfile import FileUtil

def lda_test():
    X = lda.datasets.load_reuters()
    vocab = lda.datasets.load_reuters_vocab()
    titles = lda.datasets.load_reuters_titles()
    X.shape

    model = lda.LDA(n_topics=20, n_iter=500, random_state=1)
    model.fit(X)
    topic_word = model.topic_word_  # model.components_ also works
    n_top_words = 8
    for i, topic_dist in enumerate(topic_word):
        topic_words = np.array(vocab)[np.argsort(topic_dist)][:-n_top_words:-1]
        print('Topic {}: {}'.format(i, ' '.join(topic_words)))

def load_gof_data():
    from pre_processing import Loader, PreProcessing
    loader = Loader()
    word_lst = loader.get_list('data/testdatatxt.txt')
    pre_process = PreProcessing()
    lst = pre_process.process2(word_lst)
    return lst 
   
# http://radimrehurek.com/gensim/tut1.html
def gensim():
    from gensim import corpora, models, similarities
    documents = load_gof_data()
    stoplist = set('for a of the and to in'.split())
    texts = [[word for word in document.lower().split() if word not in stoplist]
             for document in documents]

    from collections import defaultdict
    frequency = defaultdict(int)
    for text in texts:
        for token in text:
            frequency[token] += 1
    
    texts = [[token for token in text if frequency[token] > 1]
             for text in texts]
    
    #print texts
    
    from pprint import pprint  # pretty-printer
    #pprint(texts)
#     texts = ["a b c".split(), "a a b".split()]
        
    dictionary = corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    #print corpus
#     model = models.TfidfModel(corpus, id2word=dictionary)
    model = models.LdaModel(corpus, num_topics=50)
#     model = models.LsiModel(corpus, id2word=dictionary, num_topics=3)
    
    doc = 'report data'
    vec_bow = dictionary.doc2bow(doc.lower().split())
#     print vec_bow
    vec_space = model[vec_bow]
    print 'vector ', vec_space
#     index = similarities.MatrixSimilarity(model[corpus])
#     sims = index[vec_space]
#     print list(enumerate(sims))
    
#     print index
#     print vec_space
#     print vec_bow
#     vec_bow = dictionary.doc2bow(doc.lower().split())
#     vec_lsi = lsi[vec_bow]
#     index = similarities.MatrixSimilarity(lsi[corpus])
#     sim = index[lsi[vec_lsi]]
#     print sim
#     print vec_lsi
    
#     index = similarities.MatrixSimilarity(lsi[corpus])
#     print list(enumerate(index))
    
#     print lsi
#     print corpus
#     print dictionary
#     lsi = models.LsiModel()
#     print corpus
#     tfidf = models.TfidfModel(corpus)
#     corpus_tfidf = tfidf[corpus]
#     for doc in corpus_tfidf:
#         print doc
    
#     pprint(tfidf)
#     
#     test1 = "eps trees time"
#     print test1.split()
#     print dictionary.token2id
#     
#     test_word = dictionary.doc2bow(test1.split())
#     print test_word
#     
#     print tfidf[test_word]
    #print test_word
    #print tfidf
    #print corpus
    #dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
    #print(dictionary)
    #print(dictionary.token2id)
            
if __name__ == '__main__':
    gensim()
