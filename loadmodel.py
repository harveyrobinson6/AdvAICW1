#print all runs to console
#have user input number corresponding to array index of run
#load that model and ask user for text input to predict

import os
import pickle
import re
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

RUNS_DIR = "saves/runs"


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
# Load model + tfidf
# -----------------------------
with open(os.path.join(run_path, "mlp_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(run_path, "tfidf.pkl"), "rb") as f:
    tfidf = pickle.load(f)

print("Model loaded successfully.\n")
techs = format_nlp_techniques(nlp_config)
print("This model uses the following NLP techniques:", techs)

# -----------------------------
# Interactive prediction
# -----------------------------
while True:

    text = input("Enter text (or type quit): ")

    if text.lower() == "quit":
        break

    clean = preprocess_text(
    text,
    lowercase=nlp_config["lowercase"],
    remove_urls=nlp_config["remove_urls"],
    remove_stopwords=nlp_config["remove_stopwords"],
    lemmatize=nlp_config["lemmatization"],
    stem=nlp_config["stemming"]
)

    vec = tfidf.transform([clean])

    prediction = model.predict(vec)[0]

    print("Input after processing: ", clean)
    print("Prediction: ", prediction)
    print()