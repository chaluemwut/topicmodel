# -*- coding: utf-8 -*-
from gensim import corpora, models, similarities
from pre_processing import Loader, PreProcessing
import printutil, pickle

def tf_idf(documents, doc_test):
#     documents = [u'การ ทำงาน',u'ทดสอบ ใช้']
    texts = [[word for word in document.split()] for document in documents]
#     printutil.printUtilL2(texts)
    dictionary =  corpora.Dictionary(texts)
    corpus = [dictionary.doc2bow(text) for text in texts]
    model = models.TfidfModel(corpus, id2word=dictionary)

#     doc = "การ ทำ งาน"
#     printutil.printUtilL1(doc.split())
    vec_bow = dictionary.doc2bow(doc_test.split())
    vec_space = model[vec_bow]
    index = similarities.MatrixSimilarity(model[corpus])
    sims = index[vec_space]
    
    sims = sorted(enumerate(sims), key=lambda item: -item[1])
    return sims, dictionary

    
if __name__ == '__main__':
    documents = pickle.load(open('data/obj/filter.obj'))
    agree_filter = pickle.load(open('data/obj/agree_filter.obj'))
    for doc_test in agree_filter:
        sim_obj, dictionary = tf_idf(documents, doc_test)
        ind = sim_obj[0][0]
        dic_token = dictionary.token2id
        print dic_token[1]
#     tf_idf(documents, agree_filter[1])
# #     print documents[94]
#     print agree_filter[4]
