# Cascading Tweet Cyberbullying Detection System

A machine learning-based application designed to detect and categorize cyberbullying in tweet text data. The system utilizes a cascading inference architecture, first identifying the presence of bullying (binary) and subsequently classifying the specific type of abuse (multi-class).

This application was built for a Data Science Final Project presented by **Group 3 (Aletheia)**.

## Features

* **Cascading Inference Pipeline:** Two-stage model execution for efficient and accurate filtering.
* **Robust Preprocessing:** Integrated cleaning and lemmatization pipeline using NLTK.
* **Advanced Feature Engineering:** Uses Word2Vec embeddings weighted by TF-IDF scores for superior semantic representation.
* **Interactive UI:** Built with Streamlit for real-time text analysis.
* **Explainability:** Debugging view to visualize processed text and label detection.

## Architecture

The system follows a two-stage classification logic:

1. **Stage 1 (Binary Model):** Determines if the input text is `cyberbullying` or `not_cyberbullying`.
2. **Stage 2 (Multiclass Model):** Only if *Stage 1* is positive, it classifies the bullying into specific categories such as:
    * `age`
    * `ethnicity`
    * `gender`
    * `religion`
    * `other_cyberbullying`

## Tech Stack

* **Machine Learning:** [LightGBM](https://lightgbm.readthedocs.io/), [Scikit-Learn](https://scikit-learn.org/), [Gensim](https://radimurek.github.io/gensim/)
* **Frontend:** [Streamlit](https://streamlit.io/)
* **NLP:** [NLTK](https://www.nltk.org/)
* **Data Handling:** NumPy, Pandas

## Setup and Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed. You will need to download the model artifacts (`.pkl` and `.gensim` files) and place them in the project root directory.

### 2. Clone the Repository

```bash
git clone https://github.com
cd your-repo-name
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Running the Application

```bash
streamlit run app.py
```

## Project Structure

```text
├── app.py              # Streamlit frontend application
├── inference.py        # Core logic for preprocessing and prediction
├── requirements.txt    # Project dependencies
├── lgbm_binary.pkl     # Binary model artifact
├── lgbm_multiclass.pkl # Multiclass model artifact
├── w2v_model.gensim    # Word2Vec word embeddings
├── tfidf_dict.pkl      # TF-IDF weights dictionary
└── README.md
```

## Dataset Reference

This project was trained on a [Kaggle Cyberbullying Classification Dataset](https://www.kaggle.com/datasets/andrewmvd/cyberbullying-classification).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
