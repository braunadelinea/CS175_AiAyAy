import re
import math
import numpy as np
from collections import Counter

def process_documents(documents):
    processed_docs = []
    for document in documents:
        processed_doc = []
        for term in document.split():
            processed_doc.append(re.sub(r"[^a-zA-Z']", '', term).lower())
        processed_docs.append(processed_doc)
    return processed_docs

def get_terms_in_corpus(documents):
    term_dictionary = {}
    for document in documents:
        for term in document:
            if term not in term_dictionary:
                term_dictionary[term] = 1
            else:
                term_dictionary[term] += 1
    term_list = []
    for term in term_dictionary:
        if term_dictionary[term] > 5:
            term_list.append(term)
    return term_list

def calculate_tf_idf(documents, terms):
    tf_idfs = {}
    doc_counters = [Counter(doc) for doc in documents]
    doc_freqs = Counter()
    for doc_counter in doc_counters:
        doc_freqs.update(doc_counter.keys())
    
    N = len(documents)
    
    for i, term in enumerate(terms):
        df = doc_freqs[term]
        if df > 0:
            idf = math.log(N / df, 10)
            for j, doc_counter in enumerate(doc_counters):
                count = doc_counter[term]
                if count > 0:
                    tf = 1 + math.log(count, 10)
                    tf_idfs[(i, j)] = tf * idf
    return tf_idfs
