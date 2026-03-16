import os
import datamanager
from collections import Counter

def load_data():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    detection_path = os.path.join(base_dir, "Data/language_detection_dataset.csv")
    identification_path = os.path.join(base_dir, "Data/language_identification_dataset.csv")
                
    df = datamanager.load_and_combine(detection_path, identification_path)
    df = datamanager.basic_clean_and_filter(df, min_chars=30, max_chars=600)
    
    return df['text'].tolist()

def generate_char_ngrams(text, n_min=3, n_max=5):
    text = text.lower()
    ngrams = []
    length = len(text)
    for n in range(n_min, n_max + 1):
        for i in range(length - n + 1):
            ngrams.append(text[i : i + n])
    return ngrams

def process_documents(documents):
    processed_docs = []
    for document in documents:
        ngrams = generate_char_ngrams(document, n_min=3, n_max=5)
        processed_docs.append(ngrams)
    return processed_docs

def get_terms_in_corpus(processed_docs, min_df=2):
    term_doc_freq = Counter()
    for doc_ngrams in processed_docs:
        unique_ngrams = set(doc_ngrams)
        term_doc_freq.update(unique_ngrams)
        
    term_list = [term for term, count in term_doc_freq.items() if count >= min_df]
    return term_list

def calculate_ngram_counts(processed_docs, terms):
    matrix = {}
    
    term_to_idx = {term: i for i, term in enumerate(terms)}
    
    for j, doc_ngrams in enumerate(processed_docs):
        doc_counter = Counter(doc_ngrams)
        for term, count in doc_counter.items():
            if term in term_to_idx:
                i = term_to_idx[term]
                matrix[(i, j)] = count
                
    return matrix
