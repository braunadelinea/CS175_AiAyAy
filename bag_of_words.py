import re
import os
import datamanager
from collections import Counter

def load_data():
    df = datamanager.load_and_combine("Data/language_detection_dataset.csv", "Data/language_identification_dataset.csv")
    df = datamanager.basic_clean_and_filter(df, min_chars=30, max_chars=600)
    return df['text'].tolist()

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

def calculate_bag_of_words(documents, terms):
    bow_matrix = {}
    doc_counters = [Counter(doc) for doc in documents]
    
    for i, term in enumerate(terms):
        for j, doc_counter in enumerate(doc_counters):
            count = doc_counter[term]
            if count > 0:
                bow_matrix[(i, j)] = count
    return bow_matrix
