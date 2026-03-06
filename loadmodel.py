#print all runs to console
#have user input number corresponding to array index of run
#load that model and ask user for text input to predict

import os
import pickle
import re
import pandas as pd
import sys
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from tensorflow import keras
from tensorflow.keras.preprocessing.sequence import pad_sequences

RUNS_DIR = ""
MLP_DIR = "saves/mlp/runs"
CNN_DIR = "saves/cnn/runs"


# -----------------------------
# Preprocessing (same as training)
# -----------------------------
def preprocess_text(text, lowercase, remove_urls, remove_stopwords, lemmatize, stem):

    text = str(text)

    if lowercase:
        text = text.lower()

    if remove_urls:
        text = re.sub(r'http\S+|www\S+', '', text)

    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    if remove_stopwords:
        stop_words = set(stopwords.words("english"))
        tokens = [t for t in tokens if t not in stop_words]

    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(t) for t in tokens]

    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(t) for t in tokens]

    return " ".join(tokens)

def load_nlp_config(path):
    config = {}

    with open(path, "r") as f:
        for line in f:
            key, value = line.strip().split("=")
            config[key] = value.lower() == "true"

    return config

def format_nlp_techniques(config):
    pretty_names = {
        "lowercase": "Lowercasing",
        "remove_urls": "URL Removal",
        "remove_stopwords": "Stopword Removal",
        "lemmatization": "Lemmatization",
        "stemming": "Stemming"
    }

    enabled = [pretty_names[k] for k, v in config.items() if v]

    if not enabled:
            return "None"


    return ", ".join(enabled)


# -----------------------------
# List runs
# -----------------------------

while True:
    choice = input("\n'mlp' or 'cnn' ")
    if choice == "mlp":
        RUNS_DIR = MLP_DIR
        break
    elif choice == "cnn":
        RUNS_DIR = CNN_DIR
        break
    else:
        print("Input 'mlp' or 'cnn'")

runs = sorted(os.listdir(RUNS_DIR))

if len(runs) == 0:
    print("No saved runs found.")
    exit()

print("\nAvailable Runs:\n")

for i, run in enumerate(runs):
    print(f"{i} : {run}")

# -----------------------------
# Select run
# -----------------------------
while True:
    try:
        choice = int(input("\nSelect run number: "))
        if 0 <= choice < len(runs):
            break
        else:
            print("Invalid number.")
    except ValueError:
        print("Enter a valid integer.")

selected_run = runs[choice]
run_path = os.path.join(RUNS_DIR, selected_run)

config_path = os.path.join(run_path, "nlpconfig.txt")
nlp_config = load_nlp_config(config_path)

print("\nLoading run:", selected_run)

# -----------------------------
# Load model
# -----------------------------
if RUNS_DIR == MLP_DIR:

    with open(os.path.join(run_path, "mlp_model.pkl"), "rb") as f:
        model = pickle.load(f)

    with open(os.path.join(run_path, "tfidf.pkl"), "rb") as f:
        tfidf = pickle.load(f)

    model_type = "mlp"

else:

    model = keras.models.load_model(os.path.join(run_path, "cnn_model.keras"))

    with open(os.path.join(run_path, "tokenizer.pkl"), "rb") as f:
        tokenizer = pickle.load(f)

    MAX_LEN = 200
    model_type = "cnn"

print("Model loaded successfully.\n")
techs = format_nlp_techniques(nlp_config)
print("This model uses the following NLP techniques:", techs)

# -----------------------------
# Interactive prediction
# -----------------------------
while True:

    print("\nInput method:")
    print("1 - Enter text manually")
    print("2 - Load CSV file")
    print("exit - exit")

    choice = input("Select option: ")

    if choice == "1":
        while True:

            headline = input("\nHeadline (optional): ")
            post = input("Post text: ")

            if post.lower() == "exit":
                break

            combined = headline + " " + post

            clean_text = preprocess_text(
                combined,
                nlp_config["lowercase"],
                nlp_config["remove_urls"],
                nlp_config["remove_stopwords"],
                nlp_config["lemmatization"],
                nlp_config["stemming"]
            )

            if model_type == "mlp":

                vec = tfidf.transform([clean_text])
                prediction = model.predict(vec)[0]

            else:

                seq = tokenizer.texts_to_sequences([clean_text])
                pad = pad_sequences(seq, maxlen=MAX_LEN)

                prob = model.predict(pad)[0][0]
                prediction = int(prob > 0.5)

            print("Input after processing:", clean_text)
            print("Prediction:", prediction)
            print()

    elif choice == "2":

        csv_path = input("Enter CSV file path: ")

        df = pd.read_csv(csv_path)

        # ensure columns exist
        if "post" not in df.columns:
            print("CSV must contain a 'post' column.")
            exit()

        if "news_headline" not in df.columns:
            df["news_headline"] = ""

        df["text"] = df["news_headline"].fillna("") + " " + df["post"].fillna("")

        df["clean_text"] = df["text"].apply(
            lambda x: preprocess_text(
                x,
                nlp_config["lowercase"],
                nlp_config["remove_urls"],
                nlp_config["remove_stopwords"],
                nlp_config["lemmatization"],
                nlp_config["stemming"]
            )
        )

        # ---------- MLP ----------
        if model_type == "mlp":

            vec = tfidf.transform(df["clean_text"])
            preds = model.predict(vec)

            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(vec)[:,1]
            else:
                probs = preds

        # ---------- CNN ----------
        else:

            seq = tokenizer.texts_to_sequences(df["clean_text"])
            pad = pad_sequences(seq, maxlen=MAX_LEN)

            probs = model.predict(pad).flatten()
            preds = (probs > 0.5).astype(int)

        df["misinfo_probability"] = probs
        df["prediction"] = preds

        output_path = "predictions.csv"
        df.to_csv(output_path, index=False)

        print(f"\nPredictions saved to {output_path}")

    elif choice == "exit":
        sys.exit()

    else:
        print("Invalid Input")