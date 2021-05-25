import fitz
from collections import defaultdict
import string

def get_titles(pdf, start, end, stop, value):
    words_blocks = []
    for page in range(0,len(pdf),1):
        block_ = [pdf[page].getTextPage().extractWORDS()]
        b = [x for y in block_ for x in y]
        words_blocks.append([b, page+1])
        
    words_ = []
    for block in words_blocks:
        for b in block[0]:
            words_.append([round(fitz.Rect(b[:4]).height,2), b[4], block[1], b[5]])
                       
    titles_words = [item[1:] for item in words_ if item[0] == value]
    
    titles_ = defaultdict(list)
    
    for w in titles_words: 
        titles_[w[1], w[-1]].append(w[0])
    for k in titles_.keys():
        titles_[k] = ' '.join(titles_[k])
        
    
    titles = dict(titles_)    
        
    titles_ = {key:val for key, val in titles.items() if val != 'CONCLUSION'}
    deletions = ['Article', 'Amendment', 'CONCLUSION', 'MAP ', 'CHAPTER OUTLINE']
    for d in deletions:
        titles_ = {key:val for key, val in titles_.items() if not val.startswith(d)}
        
        
    return titles_ 
            
    