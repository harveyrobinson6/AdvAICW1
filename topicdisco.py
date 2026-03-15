'''

100285823 - Latent Dirichlet Allocation

PIPELINE:

Dataset
Preprocessing
NLU Intent Extraction
Vectorisation (BOW / TF-IDF)
LDA Topic Modelling
Assign topic to each document
Topic analysis
Metrics and graphs

'''

import pandas as pd
import sys
import os
import pickle
import spacy
from collections import Counter
from datetime import datetime
import datastats
import preprocessing
import matplotlib.pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

nlp = spacy.load("en_core_web_sm")

def load_data(path):
    df = pd.read_csv(path)
    return df

#extract top words per topic
def get_topics(model, vectorizer, top_n=10):

    #get vocabulary words from the vectorizer
    words = vectorizer.get_feature_names_out()
    topics = []

    #loop through each discovered topic
    #select top N words with highest probability in this topic
    #store topic id and its important words
    for topic_idx, topic in enumerate(model.components_):
        top_words = [words[i] for i in topic.argsort()[-top_n:]]
        topics.append((topic_idx, top_words))

    return topics

def print_topics(model, vectorizer, top_n=10):

    #get vocabulary words from the vectorizer
    words = vectorizer.get_feature_names_out()

    #loop through each topic
    #get top words for this topic
    #print topic summary
    for topic_idx, topic in enumerate(model.components_):

        top_words = [words[i] for i in topic.argsort()[-top_n:]]

        print(f"Topic {topic_idx}: {' '.join(top_words)}")

def extract_intents(text):

    doc = nlp(text)

    intents = []

    for token in doc:

        #look for direct object dependency (verb-object relationship)
        if token.dep_ == "dobj":

            #get the verb connected to this object
            verb = token.head.text
            objs = [token.text]

            #handle multiple objects like "pizza and cola"
            for child in token.children:
                if child.dep_ == "conj":
                    objs.append(child.text)

            #create verb-object intent pairs
            for obj in objs:
                intents.append(f"{verb}_{obj}")

    return intents

if __name__ == "__main__":

    DATA_PATH = "social-media-release.csv"
    RUNS_DIR = "saves/lda/runs"

    os.makedirs(RUNS_DIR, exist_ok=True)

    #ask user for run name
    run_name = input("Enter a name for this run: ").strip()
    #lower, urls, stopwords, lemmatizer, stemming
    techniques = ["lowercase", "remove urls", "remove stopwords", "lemmatize", "stem"]
    results = [False, False, False, False, False]
    for x in range(len(techniques)):
        while True:
            answer = input(techniques[x] + " (y/n): ").strip().lower()

            if answer == "y":
                results[x] = True
                break
            elif answer == "n":
                results[x] = False
                break
            else:
                print("Invalid input. Please type 'y' or 'n'.")

    #lem and stem at the same time produces issues and poor results
    #hard coded so only one can be selected
    if results[3] == True and results[4] == True:
        print("Lemmatization and stemming have both been selected... defaulting to just lemmatization")
        results[4] = False

    #user selects text representation
    textrep = ""
    while True:
        answer = input("Bag of words (bow) or TF-IDF (t)?").strip().lower()

        if answer == "bow":
            textrep = "bow"
            break
        elif answer == "t":
            textrep = "t"
            break
        else:
            print("Invalid input. Please type 'bow' or 't'.")

    while True:
        answer = input("Begin?" + " (y/n): ").strip().lower()

        if answer == "y":
            break
        elif answer == "n":
            sys.exit()

    #makes a directory identified by a timestamp and name
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    RUN_DIR = os.path.join(RUNS_DIR, f"{run_id}_{run_name}")
    os.makedirs(RUN_DIR)

    #stores lda model using pickle
    LDA_PATH = os.path.join(RUN_DIR, "lda_model.pkl")

    #stores vectorizer using pickle
    VECTORIZER_PATH = ""

    if textrep == "bow":
        VECTORIZER_PATH = os.path.join(RUN_DIR, "bow.pkl")
    elif textrep == "t":
        VECTORIZER_PATH = os.path.join(RUN_DIR, "tfidf.pkl")

    #records topics
    TOPICS_PATH = os.path.join(RUN_DIR, "topics.txt")

    #records results of nlu
    NLU_RESULTS_PATH = os.path.join(RUN_DIR, "nlu_topics.txt")

    #records which nlp techniques were used
    NLP_CONFIG_PATH = os.path.join(RUN_DIR, "nlpconfig.txt")

    #records the lda config
    LDA_CONFIG_PATH = os.path.join(RUN_DIR, "lda_config.txt")

    #records metrics of the run
    METRICS_PATH = os.path.join(RUN_DIR, "metrics.txt")

    #records information about the dataset
    DATA_INFO_PATH = os.path.join(RUN_DIR, "datainfo.txt")

    #records topic results
    TOPIC_RESULTS_PATH = os.path.join(RUN_DIR, "topic_results.txt")

    #saves which nlp techniques were used
    with open(NLP_CONFIG_PATH, "w") as f:
        f.write("lowercase=" + str(results[0]) + "\n")
        f.write("remove_urls=" + str(results[1]) + "\n")
        f.write("remove_stopwords=" + str(results[2]) + "\n")
        f.write("lemmatization=" + str(results[3]) + "\n")
        f.write("stemming=" + str(results[4]) + "\n")
        
    #dataset loaded into pandas dataframe
    df = load_data(DATA_PATH)

    #finds and writes info about the dataset
    datastats.dataset_overview(df)
    datastats.missing_values(df)
    datastats.class_distribution(df)
    datastats.text_statistics(df, DATA_INFO_PATH)

    #post content is default input text
    df["post"] = df["post"].fillna("")
    df["news_headline"] = df["news_headline"].fillna("")
    
    df["text"] = df["news_headline"] + " " + df["post"]

    #applies preprocessing to every row
    df["clean_post"] = df["text"].apply(
    lambda x: preprocessing.preprocess_text(
        x,
        lowercase=results[0],
        removeurls=results[1],
        removestopwords=results[2],
        lemmatize=results[3],
        stem=results[4]
        )
    )

    texts = df["clean_post"]

    print("\nRunning NLU analysis...")

    #df["headline_intents"] = df["news_headline"].apply(extract_intents)
    #df["post_intents"] = df["post"].apply(extract_intents)

    headline_docs = list(nlp.pipe(df["news_headline"], batch_size=64))
    post_docs = list(nlp.pipe(df["post"], batch_size=64))

    df["headline_intents"] = [[f"{t.head.lemma_}_{t.lemma_}" 
                            for t in doc if t.dep_ == "dobj"]
                            for doc in headline_docs]

    df["post_intents"] = [[f"{t.head.lemma_}_{t.lemma_}" 
                        for t in doc if t.dep_ == "dobj"]
                        for doc in post_docs]

    all_intents = []

    for intents in df["headline_intents"]:
        all_intents.extend(intents)

    for intents in df["post_intents"]:
        all_intents.extend(intents)

    intent_counts = Counter(all_intents).most_common(30)

    with open(NLU_RESULTS_PATH, "w") as f:
        for intent, count in intent_counts:
            f.write(f"{intent}: {count}\n")

    print("\nTop NLU relations:")
    for intent, count in intent_counts[:10]:
        print(intent, count)

    #vectorization

    #bag of words
    if textrep == "bow":

        vectorizer = CountVectorizer(
            max_features=15000,
            min_df=5,
            max_df=0.9,
            token_pattern=r'\b[a-zA-Z]{3,}\b'
        )

    #tf-idf
    else:

        vectorizer = TfidfVectorizer(
            max_features=15000,
            min_df=5,
            max_df=0.9
        )

    X = vectorizer.fit_transform(texts)

    with open(VECTORIZER_PATH, "wb") as f:
        pickle.dump(vectorizer, f)

    #LDA
    N_TOPICS = 10

    lda = LatentDirichletAllocation(
        n_components=N_TOPICS,
        random_state=42,
        learning_method="batch"
    )

    lda.fit(X)

    #topic distribution per document
    topic_distribution = lda.transform(X)

    #each post gets most likely topic
    df["dominant_topic"] = topic_distribution.argmax(axis=1)

    with open(LDA_PATH, "wb") as f:
        pickle.dump(lda, f)

    #save config
    with open(LDA_CONFIG_PATH, "w") as f:
        f.write(f"topics={N_TOPICS}\n")
        f.write(f"text_representation={textrep}\n")

    topics = get_topics(lda, vectorizer)

    print("\nDiscovered Topics\n")

    with open(TOPICS_PATH, "w") as f:

        for topic_id, words in topics:

            line = f"Topic {topic_id}: {' '.join(words)}"

            print(line)
            f.write(line + "\n")

    print("\nTopic distribution across dataset:\n")

    topic_counts = df["dominant_topic"].value_counts().sort_index()

    for topic, count in topic_counts.items():
        print(f"Topic {topic}: {count} documents")

    print("\nTopic vs Class Label Distribution\n")

    #relationship between topics and fake/real labels
    topic_class = pd.crosstab(
        df["dominant_topic"],
        df["class_label"],
        normalize="index"
    )

    print(topic_class)

    print("\nExample headlines per topic\n")

    for topic in range(N_TOPICS):

        print(f"\nTopic {topic} examples:")

        examples = df[df["dominant_topic"] == topic]["news_headline"].drop_duplicates().head(5)

        for e in examples:
            print("-", e)

    with open(TOPIC_RESULTS_PATH, "w") as f:

        f.write("Topic distribution\n")

        for topic, count in topic_counts.items():
            f.write(f"Topic {topic}: {count} documents\n")

        f.write("\nTopic vs class label\n")
        f.write(topic_class.to_string())

    topic_counts.plot(kind="bar")

    plt.title("Topic Distribution")
    plt.xlabel("Topic")
    plt.ylabel("Number of Documents")

    plt.savefig(os.path.join(RUN_DIR, "topic_distribution.png"))
    plt.close()

    #metrics
    perplexity = lda.perplexity(X)

    with open(METRICS_PATH, "w") as f:
        f.write(f"Perplexity: {perplexity}\n")

    print("\nModel Perplexity:", perplexity)