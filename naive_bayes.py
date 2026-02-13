import os
import sys
import time
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from datamanager import load_and_combine, basic_clean_and_filter, stratified_split
import bag_of_words
import n_gram
import tf_idf

def run_naive_bayes_pipeline():
    start_time = time.time()
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(current_dir, "Data")
    
    print("Loading and combining datasets...")
    detection_path = os.path.join(data_dir, "language_detection_dataset.csv")
    identification_path = os.path.join(data_dir, "language_identification_dataset.csv")
    df = load_and_combine(detection_path, identification_path)

    print("Cleaning and filtering...")
    df = basic_clean_and_filter(df, min_chars=30, max_chars=600)

    print("Creating stratified split...")
    train_df, val_df, test_df = stratified_split(df, seed=0)
    
    print(f"Train size: {len(train_df)} | Val size: {len(val_df)} | Test size: {len(test_df)}")

 
    tokenizers = {
        "n-gram": n_gram,
        "Bag of Words": bag_of_words,
        "TF-IDF": tf_idf
    }

    from scipy.sparse import dok_matrix, csr_matrix

    for name, module in tokenizers.items():
        if module is None: 
            continue
            
        print(f"\n\n---Running Naive Bayes with {name} Tokenization---")
        
        print(f"Tokenizing {name}...")
        
        t0 = time.time()
        
        train_docs_processed = module.process_documents(train_df["text"].tolist())
        val_docs_processed = module.process_documents(val_df["text"].tolist())
        test_docs_processed = module.process_documents(test_df["text"].tolist())
        
        if name == "n-gram":
             terms = module.get_terms_in_corpus(train_docs_processed, min_df=2)
        else:
             terms = module.get_terms_in_corpus(train_docs_processed)
             
        term_map = {t: i for i, t in enumerate(terms)}
        V = len(terms)
        print(f"Vocabulary size: {V}")
        
        def vectorize(processed_docs, module_name):
            
            N = len(processed_docs)
            
            if module_name == "TF-IDF":
                matrix_dict = module.calculate_tf_idf(processed_docs, terms)
            elif module_name == "Bag of Words":
                matrix_dict = module.calculate_bag_of_words(processed_docs, terms)
            elif module_name == "n-gram":
                matrix_dict = module.calculate_ngram_counts(processed_docs, terms)
            
            rows = []
            cols = []
            data = []
            for (term_idx, doc_idx), val in matrix_dict.items():
                rows.append(doc_idx)
                cols.append(term_idx)
                data.append(val)
            
            mat = csr_matrix((data, (rows, cols)), shape=(N, V), dtype=np.float64)
            return mat

        print("Vectorizing Train...")
        X_train = vectorize(train_docs_processed, name)
        print("Vectorizing Val...")
        X_val = vectorize(val_docs_processed, name)
        print("Vectorizing Test...")
        X_test = vectorize(test_docs_processed, name)
        
        print(f"Preprocessing time: {time.time() - t0:.2f} s")

        print(f"Training MultinomialNB with {name} features...")
        nb_model = MultinomialNB()
        nb_model.fit(X_train, train_df["label"])
        
        print("Evaluating...")
        val_pred = nb_model.predict(X_val)
        test_pred = nb_model.predict(X_test)

        print(f"Validation accuracy: {accuracy_score(val_df['label'], val_pred):.4f}")
        print(f"Test accuracy:       {accuracy_score(test_df['label'], test_pred):.4f}")
        
        print("Classification Report (Test):")
        print(classification_report(test_df["label"], test_pred, digits=4, zero_division=0))

    total_time = time.time() - start_time
    print(f"\nTotal runtime: {total_time:.2f} seconds")

def main():
    print("-----Running Naive Bayes Model-----\n")
    run_naive_bayes_pipeline()

if __name__ == "__main__":
    main()
