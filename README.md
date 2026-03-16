
External Libraries Used:
- pandas (https://pandas.pydata.org/)
- numpy (https://numpy.org/)
- scikit-learn (https://scikit-learn.org/)
- PyTorch (https://pytorch.org/)
- Hugging Face Transformers (https://huggingface.co/docs/transformers/)
- Hugging Face Datasets (https://huggingface.co/docs/datasets/)
- Hugging Face Hub (https://huggingface.co/docs/huggingface_hub/)
- librosa (https://librosa.org/)

Publicly available codes used: 
- Hugging Face Transformers examples and documentation
  https://huggingface.co/docs/transformers/
  Used as reference for loading pretrained models and building the BERT and Gemma inference pipelines. Modified/added approximately 80 lines of code.
- Scikit-learn documentation examples
  https://scikit-learn.org/stable/
  Used as reference for implementing Logistic Regression, Naive Bayes, and Random Forest classifiers.

Scripts written by our team: 
- bert_language_classifier.py (~150 lines)
  Implements a multilingual BERT model using the Hugging Face Transformers library. Handles tokenization, dataset preparation, model fine-tuning, and evaluation
- llm_gemma.py (~140 lines)
  Implements a zero-shot language classification baseline using the Gemma 3-1B-IT large language model. Constructs prompts, runs inference, and parses generated outputs to compute classification accuracy.
  
