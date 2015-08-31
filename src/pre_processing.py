from nlp import CRFWordSegment

class Loader(object):

    def __init__(self):
        pass
    
    def get_list(self, file_name):
        with open(file_name) as f:
            lines = f.readlines()
            return lines
    
class PreProcessing(object):
    
    def remove_bank_word(self, word_lst):        
        return [x for x in word_lst]
    
    def remove_dup_sentense(self, word_list):
        from sets import Set
        ret = Set()
        for l in word_list:
            ret.add(l)
        return list(ret)
    
    def process(self, word_list):
        ret = []
        lst = self.remove_dup_sentense(word_list)
        crf = CRFWordSegment()
        for l in lst:
            ret.append(crf.crfpp(unicode(l,'utf8'))[0])
        return ret

if __name__ == '__main__':
    obj = Loader()
    words = obj.get_list('filter.txt')
     
    preProcess = PreProcessing()
    lst = preProcess.process(words)
    for l in lst:
        for ll in l:
            print ll
        print '************'    
#     print ret[0][0], ret[0][1]
#     word_pro = preProcess.remove_dup_sentense(words)
#     for x in word_pro:
#         print word_pro.count(x) , x               
#    [print x for x in obj.get_list('filter.txt')]
    
        