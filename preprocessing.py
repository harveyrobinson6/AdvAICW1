import pandas as pd
import numpy as np
import re
import os
import pickle
import io
from datetime import datetime
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



#nltk.download("punkt")
#nltk.download("punkt_tab")
#nltk.download("stopwords")
#nltk.download("wordnet")
#nltk.download('wordnet')    
#nltk.download('omw-1.4') 
#nltk.download('averaged_perceptron_tagger_eng') 

# -----------------------------
# Load data
# -----------------------------
def load_data(path):
    df = pd.read_csv(path)
    return df


# -----------------------------
# Basic dataset overview
# -----------------------------
def dataset_overview(df):
    print("Number of documents:", len(df))
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nData types:")
    print(df.dtypes)


# -----------------------------
# Missing value analysis
# -----------------------------
def missing_values(df):
    print("\nMissing values per column:")
    print(df.isnull().sum())


# -----------------------------
# Class balance
# -----------------------------
def class_distribution(df):
    counts = df["class_label"].value_counts()
    proportions = df["class_label"].value_counts(normalize=True)

    print("\nClass distribution (counts):")
    print(counts)

    print("\nClass distribution (proportions):")
    print(proportions)


# -----------------------------
# Text length statistics
# -----------------------------
def text_statistics(df, text_column="post"):
    lengths = df[text_column].astype(str).apply(len)

    print("\nText length statistics:")
    print("Min length:", lengths.min())
    print("Max length:", lengths.max())
    print("Mean length:", round(lengths.mean(), 2))
    print("Median length:", lengths.median())


def preprocess_text(text):
    parsed = str(text)
    # 1. lowercase
    lower = parsed.lower()
    # 2. remove urls / punctuation
    withouturls = re.sub(r'http\S+|www\S+', '', lower)
    # 3. tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(withouturls)
    # 4. remove stopwords
    custom_stopwords = {"read", "more", "click", "share"}
    stop_words = set(stopwords.words("english")).union(custom_stopwords)
    #stop_words = set(stopwords.words('english'))
    filtered_tokens = [word for word in tokens if word not in stop_words]
    # 5. lemmatize or stem
    lemmatizer = WordNetLemmatizer()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in filtered_tokens]

    #porter_stemmer = PorterStemmer()
    #stemmed_words = [porter_stemmer.stem(word) for word in lemmatized_words]
    # 6. return cleaned string
    #print(lemmatized_words)
    return " ".join(lemmatized_words)

if __name__ == "__main__":
    DATA_PATH = "social-media-release.csv"
    TFIDF_DIR = "saves/current_tfidf"
    RUNS_DIR = "saves/runs"

    os.makedirs(TFIDF_DIR, exist_ok=True)
    os.makedirs(RUNS_DIR, exist_ok=True)

    TFIDF_PATH = os.path.join(TFIDF_DIR, "tfidf_vectoriser.pkl")

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_DIR = os.path.join(RUNS_DIR, run_id)
    os.makedirs(RUN_DIR)

    MLP_PATH = os.path.join(RUN_DIR, "mlp_model.pkl")
    CONFIG_PATH = os.path.join(RUN_DIR, "config.txt")
    THIS_RUN_TFIDF_PATH = os.path.join(RUN_DIR, "tfidf.pkl")


    # -----------------------------
    # Load & inspect data
    # -----------------------------
    df = load_data(DATA_PATH)

    dataset_overview(df)
    missing_values(df)
    class_distribution(df)
    text_statistics(df)

    # -----------------------------
    # Preprocess text
    # -----------------------------
    df["clean_post"] = df["post"].apply(preprocess_text)

    X = df["clean_post"]
    y = df["class_label"]

    # -----------------------------
    # Train / val / test split
    # -----------------------------
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    # -----------------------------
    # TF-IDF: load or train
    # -----------------------------
    if os.path.exists(TFIDF_PATH):
        print("Loading existing TF-IDF vectoriser...")
        with open(TFIDF_PATH, "rb") as f:
            tfidf = pickle.load(f)
    else:
        print("Training TF-IDF vectoriser...")
        tfidf = TfidfVectorizer(
            max_features=8000,
            min_df=5,
            max_df=0.9,
            ngram_range=(1, 2),
            sublinear_tf=True
        )
        tfidf.fit(X_train)

        with open(TFIDF_PATH, "wb") as f:
            pickle.dump(tfidf, f)

        with open(THIS_RUN_TFIDF_PATH, "wb") as f:
            pickle.dump(tfidf, f)


    # Vectorise (always fast)
    X_train_vec = tfidf.transform(X_train)
    X_val_vec   = tfidf.transform(X_val)
    X_test_vec  = tfidf.transform(X_test)

    print("TF-IDF train shape:", X_train_vec.shape)

    # -----------------------------
    # MLP: load or train
    # -----------------------------
    mlp = MLPClassifier(
            hidden_layer_sizes=(128,),
            activation="relu",
            solver="adam",
            learning_rate_init=0.001,
            max_iter=200,
            early_stopping=True,
            n_iter_no_change=5,
            random_state=42
        )

    mlp.fit(X_train_vec, y_train)

    with open(MLP_PATH, "wb") as f:
        pickle.dump(mlp, f)
        

    # -----------------------------
    # Validation evaluation
    # -----------------------------
    y_val_pred = mlp.predict(X_val_vec)

    output_buffer = io.StringIO()

    print("Validation accuracy:", accuracy_score(y_val, y_val_pred))
    print("Validation precision:", precision_score(y_val, y_val_pred))
    print("Validation recall:", recall_score(y_val, y_val_pred))
    print("Validation F1:", f1_score(y_val, y_val_pred))

    with open(CONFIG_PATH, "w") as f:
        f.write("MLP configuration:\n")
        f.write("hidden_layer_sizes=(128,)\n")
        f.write("learning_rate_init=0.001\n")
        f.write("early_stopping=True\n")



'''
# -----------------------------
# Main execution
# -----------------------------
if __name__ == "__main__":

    if os.path.exists("tfidf_vectoriser.pkl"):
        pass
        #do the stuff
    else:
        pass
        #load in the csv

    data_path = "social-media-release.csv"
    df = load_data(data_path)

    dataset_overview(df)
    missing_values(df)
    class_distribution(df)
    text_statistics(df)

    #preprocess_text("This is a sample sentence showing stopword removal.")
    #preprocess_text("BREAKING!!! 🚨🚨 Americans could lose their housing as eviction moratorium ends 😡 Read more here: https://t.co/abc123#HousingCrisis #COVID19")
    df["clean_post"] = df["post"].apply(preprocess_text)

    # Features and labels
    X = df["clean_post"]
    y = df["class_label"]

    # First split: train vs temp (val + test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        stratify=y,
        random_state=42
    )

    # Second split: validation vs test
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        stratify=y_temp,
        random_state=42
    )

    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    tfidf = TfidfVectorizer(
        max_features=10000,
        ngram_range=(1, 2)
    )

    X_train_vec = tfidf.fit_transform(X_train)
    X_val_vec   = tfidf.transform(X_val)
    X_test_vec  = tfidf.transform(X_test)

    print("TF-IDF train shape:", X_train_vec.shape)

    #np.save("X_train.npy", X_train_vec.toarray())
    #np.save("X_val.npy",   X_val_vec.toarray())
    #np.save("X_test.npy",  X_test_vec.toarray())

    np.save("y_train.npy", y_train.values)
    np.save("y_val.npy",   y_val.values)
    np.save("y_test.npy",  y_test.values)

    mlp = MLPClassifier(
        hidden_layer_sizes=(128,),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=20,
        random_state=42
    )

    mlp.fit(X_train_vec, y_train)

    y_val_pred = mlp.predict(X_val_vec)

    val_accuracy = accuracy_score(y_val, y_val_pred)
    val_precision = precision_score(y_val, y_val_pred)
    val_recall = recall_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred)

    print("Validation accuracy:", val_accuracy)
    print("Validation precision:", val_precision)
    print("Validation recall:", val_recall)
    print("Validation F1:", val_f1)
'''