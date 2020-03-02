import tkinter as tk

import json
import pickle
import operator
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from Indexer import Posting
import re
from bs4 import BeautifulSoup
import numpy as np
import collections

## Used as reference for cosine similarity
## https://masongallo.github.io/machine/learning,/python/2016/07/29/cosine-similarity.html
def cos_sim(a, b):
	"""Takes 2 vectors a, b and returns the cosine similarity according 
	to the definition of the dot product
	"""
	dot_product = np.dot(a, b)
	norm_a = np.linalg.norm(a)
	norm_b = np.linalg.norm(b)
	return dot_product / (norm_a * norm_b)

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








def run(userInput):   
    user_input = tokenize(userInput)
    if(len(user_input) == 1):
        results = inverted_index[user_input[0]]
        count = 1
        for result in results[:10]:
            docID = result.docID
            print(count, data[docID])
            count += 1
        return results[:10]
    else:
        cos_results = []
        count = 1
        compare_set_dict = dict()
        for token in user_input:
            results = inverted_index[token][:50] ## Index elmination technique from week 7 efficient scoring lecture
            for posting in results:
                if token not in compare_set_dict:  
                    compare_set_dict[token] = set()
                    compare_set_dict[token].add(posting.docID)
                else:
                    compare_set_dict[token].add(posting.docID)
        intersection = [*compare_set_dict.values()]
        u = set.intersection(*intersection)
        for docID in u:
            path = 'WEBPAGES_RAW/' + docID
            with open(path, 'r', encoding = 'utf-8') as source_code:
                soup = BeautifulSoup(source_code, "html5lib")
                docTerms = tokenize(soup.get_text())
                docTerms = collections.Counter(docTerms)
                copy = docTerms.copy()
                for element in copy:
                    copy[element] = 0
                for token in user_input:
                    copy[token] += 1
                vector1 = np.array([*docTerms.values()])
                vector2 = np.array([*copy.values()])
                cos_results.append((docID, cos_sim(vector1, vector2)))
        cos_results.sort(key = lambda x: -x[1])
        for result in cos_results[:10]:
            docID = result[0]
            print(count, data[docID])
            count += 1
        return cos_results[:10]

if __name__ == "__main__":
    with open('inverted_index.pickle', 'rb') as handle:
        inverted_index = pickle.load(handle)
        for word in inverted_index:
            inverted_index[word] = sorted(inverted_index[word], key = lambda i: -i.tfidf)
    with open('WEBPAGES_RAW/bookkeeping.json', 'r') as json_file:
        data = json.load(json_file)
    root= tk.Tk()

    WIDTH = 1000
    HEIGHT = 800
    canvas1 = tk.Canvas(root, width = WIDTH, height = HEIGHT,  relief = 'raised')
    canvas1.pack()

    label1 = tk.Label(root, text='Search Engine')
    label1.config(font=('helvetica', 14))
    canvas1.create_window(500, 50, window=label1)

    label2 = tk.Label(root, text='Input:')
    label2.config(font=('helvetica', 12, 'bold'))
    canvas1.create_window(385, 100, window=label2)

    entry1 = tk.Entry (root) 
    canvas1.create_window(500, 100, window=entry1)
    label4 = tk.Label(root, text= "",font=('helvetica', 12, 'bold'))
    label4.pack()
    canvas1.create_window(500, 400, window=label4)
    def search ():
        userInput = entry1.get()
        
        label3 = tk.Label(root, text= 'Results\n' + '-'*100,font=('helvetica', 10))
        canvas1.create_window(500, 170, window=label3)

        result = ""
        count = 1
        test = run(userInput)
        for post in test:
            if(type(post) == Posting):
                docID = post.docID
            else:
                docID = post[0]
            result += str(count) + ' ' + str(data[docID])+'\n'+'\n'
            path = 'WEBPAGES_CLEAN/' + docID
            count += 1
        label4.config(text = result)

        
    button1 = tk.Button(text='Search!', command=search, bg='brown', fg='black', font=('helvetica', 9, 'bold'))
    canvas1.create_window(500, 130, window=button1)
    root.mainloop()


