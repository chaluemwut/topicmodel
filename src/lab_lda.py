import numpy as np
import lda
import lda.datasets

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

# http://radimrehurek.com/gensim/tut1.html
def gensim():
    from gensim import corpora, models, similarities
    documents = ["Human machine interface for lab abc computer applications",
                 "A survey of user opinion of computer system response time",
                 "The EPS user interface management system",
                 "System and human system engineering testing of EPS",
                 "Relation of user perceived response time to error measurement",
                 "The generation of random binary unordered trees",
                 "The intersection graph of paths in trees",
                 "Graph minors IV Widths of trees and well quasi ordering",
                 "Graph minors A survey"]
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
    
    from pprint import pprint  # pretty-printer
    pprint(texts)
    dictionary = corpora.Dictionary(texts)
    dictionary.save('/tmp/deerwester.dict') # store the dictionary, for future reference
    print(dictionary)
    print(dictionary.token2id)
            
if __name__ == '__main__':
    gensim()
