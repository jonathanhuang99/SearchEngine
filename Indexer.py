import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
import json
from bs4 import BeautifulSoup
import collections
import math
import pickle

CORPUS_SIZE = 37497


class Posting:
    def __init__(self, docID, termFrequency):
        self.docID = docID
        self.termFrequency = termFrequency
        self.special = 0
        self.tfidf = 0
    def __repr__(self):
        return str(self.__dict__)

def tokenize(text: str) -> list:
    porter = PorterStemmer()
    results = []
    tokens = nltk.tokenize.word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    for word in tokens:
        word = re.sub(r'\W+', '', word.lower())
        if len(word) > 2 and word not in stop_words:
            word = porter.stem(word)
            results.append(word)
    return results

def calculateTFIDF(d: dict):
    for word in d:
        dF = len(d[word])
        for posting in d[word]:
            tf = 1 + math.log(posting.termFrequency)
            idf = math.log(CORPUS_SIZE/dF)
            TFIDF = (tf * idf) + posting.special
            posting.tfidf = TFIDF

if __name__ == "__main__":
    docCount = 0
    invertedIndex = dict()
    with open('WEBPAGES_RAW/bookkeeping.json', 'r') as json_file:
        data = json.load(json_file)
    for docID in data:
        path = 'WEBPAGES_RAW/' + docID
        with open(path, 'r', encoding = 'utf-8') as source_code:
            soup = BeautifulSoup(source_code, "html5lib")
            docTerms = tokenize(soup.get_text())
            docTerms = collections.Counter(docTerms)
            for word in docTerms:
                temp = Posting(docID, docTerms[word])
                for tags in soup.find_all(['h6', 'h5','h4','h3','h2','h1','strong','em','b','i','title']):
                    compare = set(tokenize(tags.text))
                    if word in compare:
                        temp.special += 1
                if word not in invertedIndex:
                    invertedIndex[word] = [temp]
                else:
                    invertedIndex[word].append(temp)
        print(docCount,docID)
        docCount += 1
    calculateTFIDF(invertedIndex)
    
    with open('inverted_index.pickle', 'wb') as handle:
        pickle.dump(invertedIndex, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    
