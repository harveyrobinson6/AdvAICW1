import os
import pickle
import re
import pandas as pd
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from sklearn.decomposition import LatentDirichletAllocation

RUNS_DIR = "saves/lda/runs"

def get_topic_words(model, vectorizer, top_n=10):

    words = vectorizer.get_feature_names_out()
    topics = []

    for topic in model.components_:
        top_words = [words[i] for i in topic.argsort()[-top_n:]]
        topics.append(top_words)

    return topics

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

runs = sorted(os.listdir(RUNS_DIR))

if len(runs) == 0:
    print("No saved runs found.")
    exit()

print("\nAvailable Runs:\n")

for i, run in enumerate(runs):
    print(f"{i} : {run}")

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

lda_path = os.path.join(run_path, "lda_model.pkl")

# detect which vectorizer exists
bow_path = os.path.join(run_path, "bow.pkl")
tfidf_path = os.path.join(run_path, "tfidf.pkl")

#bow_path = os.path.join(run_path, "vectorizer.pkl")
#tfidf_path = os.path.join(run_path, "vectorizer.pkl")

print("Loading model...")

with open(lda_path, "rb") as f:
    lda = pickle.load(f)

if os.path.exists(bow_path):
    with open(bow_path, "rb") as f:
        vectorizer = pickle.load(f)
    print("Using Bag-of-Words representation")

elif os.path.exists(tfidf_path):
    with open(tfidf_path, "rb") as f:
        vectorizer = pickle.load(f)
    print("Using TF-IDF representation")

else:
    print("No vectorizer found.")
    exit()

topics = get_topic_words(lda, vectorizer)

print("\nPreprocessing techniques used:")
print(format_nlp_techniques(nlp_config))

#print("\nType text to analyse topics (type 'exit' to quit)\n")

print("\n1 : Analyse text manually")
print("2 : Analyse CSV file")

choice = input("Select option: ").strip()

if choice == "1":

    print("\nType text to analyse topics (type 'exit' to quit)\n")

    while True:

        headline = input("Headline (optional): ")
        post = input("Post text: ")

        if post.lower() == "exit":
            break

        if headline.strip() == "":
            user_input = post
        else:
            user_input = headline + " " + post

        clean_text = preprocess_text(
            user_input,
            nlp_config["lowercase"],
            nlp_config["remove_urls"],
            nlp_config["remove_stopwords"],
            nlp_config["lemmatization"],
            nlp_config["stemming"]
        )

        X_input = vectorizer.transform([clean_text])
        topic_probs = lda.transform(X_input)[0]

        top_topics = topic_probs.argsort()[-3:][::-1]

        print("\nProcessed text:", clean_text)
        print("\nTop predicted topics:\n")

        for t in top_topics:
            topic_words = " ".join(topics[t])
            print(f"Topic {t}")
            print(f"Words: {topic_words}")
            print(f"Probability: {topic_probs[t]:.3f}\n")


elif choice == "2":

    csv_path = input("\nEnter CSV file path: ")

    df = pd.read_csv(csv_path)

    #mkae sure post column exists
    if "post" not in df.columns:
        print("CSV must contain a 'post' column.")
        exit()

    #headline may not exist
    if "news_headline" not in df.columns:
        df["news_headline"] = ""

    #combine text
    df["text"] = df["news_headline"].fillna("") + " " + df["post"].fillna("")

    #preprocess
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

    #vectorise
    X = vectorizer.transform(df["clean_text"])

    #get topic probabilities
    topic_probs = lda.transform(X)

    #get dominant topic
    df["dominant_topic"] = topic_probs.argmax(axis=1)

    # Store top 3 topics
    df["top_topic_1"] = topic_probs.argsort(axis=1)[:, -1]
    df["top_topic_2"] = topic_probs.argsort(axis=1)[:, -2]
    df["top_topic_3"] = topic_probs.argsort(axis=1)[:, -3]

    #save probabilities optionally
    for i in range(topic_probs.shape[1]):
        df[f"topic_{i}_prob"] = topic_probs[:, i]

    output_path = "lda_predictions.csv"
    df.to_csv(output_path, index=False)

    print(f"\nResults saved to {output_path}")