README.txt for Group AI Ay Ay

Libraries used:
    - PyTorch: Used for building and training neural networks
        (https://pytorch.org/)
    - NumPy: Used for numerical operations and array manipulation
        (https://numpy.org/)
    - Pandas: Used for dataset manipulation and dataframes
        (https://pandas.pydata.org/)
    - Scikit-Learn: Used for evaluation metrics and dataset splitting
        (https://scikit-learn.org/)
    - HuggingFace Transformers: Used to load the pretrained BERT tokenizer and model
        (https://huggingface.co/docs/transformers)
    - Gradio: Used to build a simple web interface for interacting with the model
        (https://www.gradio.app/)
    - JobLib: Used for saving and loading trained machine learning models.
        (https://joblib.readthedocs.io/)

We did not use any publicly available code.

Scripts/Functions Written Entirely by Our Team:
    - bertreport.py: Evaluates the trained language classification model on the test dataset, computing overall
      accuracy, generating a detailed classification report, producing a confusion matrix, and identifying the most common
      language misclassification pairs.
        (65 lines)

    - berttrainer.py: Trains a BERT-based language classification model on the dataset by tokenizing text, creating
      data loaders, training the model for two epochs, and saving the trained model weights.
        (88 lines)

    - interface.py: Loads the trained BERT language classification model and launches a Gradio web interface that
      predicts the top three most likely languages for user-entered text along with their confidence scores.
        (95 lines)

    - testbert.py: Loads the trained BERT language classification model and tests it by predicting the language of
      several example sentences.
        (61 lines)

    - bag_of_words.py: Implements a Bag-of-Words feature extraction pipeline by cleaning text, building a vocabulary of
      frequent terms, and generating a sparse matrix representing term frequencies across documents.
        (48 lines)

    - datamanager.py: Loads multiple language datasets, standardizes their format, cleans and filters the text data,
      fixes label inconsistencies, and provides a stratified train/validation/test split while preserving class balance.
        (109 lines)

    - full_dataset_tester.py: Evaluates the trained language classification model on an external HuggingFace language
      dataset, reporting overall accuracy, per-language performance, and how the model classifies languages it was not
      trained on.
        (138 lines)

    - llmgemma.py: Uses the Gemma language model from HuggingFace to classify the language of text samples, evaluates
      prediction accuracy against the true labels, and saves the results to a CSV file.
        (111 lines)

    - n_gram.py: Generates character n-gram features (3–5 characters) from text documents and constructs a sparse
      matrix of n-gram frequencies for use in language identification models.
        (57 lines)

    - naive_bayes.py: Runs a full Naive Bayes language classification pipeline by preprocessing the dataset, generating
      features using Bag-of-Words, character n-grams, and TF-IDF, training a Multinomial Naive Bayes model, and
      evaluating its performance on validation and test sets.
        (123 lines)

    - terminal_interface.py: Loads a trained language classification pipeline and provides a command-line interface
      that allows users to input text and receive a predicted language with a confidence score.
        (43 lines)

    - tf_idf.py: Implements TF-IDF feature extraction by cleaning text, building a vocabulary of frequent terms, and
      computing TF-IDF weights for each term across documents.
        (52 lines)

    - tk_interface.py: Creates a Tkinter graphical user interface that loads a trained language classification model
      and allows users to input text and see the predicted language along with the top three predictions and their
      confidence scores.
        (104 lines)

    - trainer.py: Trains a language classification model using TF-IDF character n-grams and Logistic Regression,
      evaluates its accuracy on validation and test sets, and saves the trained model pipeline for later use.
        (82 lines)