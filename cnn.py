import pandas as pd
import os
import io
import sys
import pickle
import datastats
import preprocessing
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def load_data(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":

    DATA_PATH = "social-media-release.csv"
    RUNS_DIR = "saves/cnn/runs"

    os.makedirs(RUNS_DIR, exist_ok=True)

    # Ask user for run name
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

    if results[3] == True and results[4] == True:
        print("Lemmatization and stemming have both been selected... defaulting to just lemmatization")
        results[4] = False

    while True:
        answer = input("Begin?" + " (y/n): ").strip().lower()

        if answer == "y":
            break
        elif answer == "n":
            sys.exit()

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    RUN_DIR = os.path.join(RUNS_DIR, f"{run_id}_{run_name}")
    os.makedirs(RUN_DIR)

    MLP_PATH = os.path.join(RUN_DIR, "mlp_model.pkl")
    CONFIG_PATH = os.path.join(RUN_DIR, "config.txt")
    NLP_CONFIG_PATH = os.path.join(RUN_DIR, "nlpconfig.txt")
    DATA_INFO_PATH = os.path.join(RUN_DIR, "datainfo.txt")
    THIS_RUN_TOKENIZER_PATH = os.path.join(RUN_DIR, "tokenizer.pkl")

    with open(NLP_CONFIG_PATH, "w") as f:
        f.write("lowercase=" + str(results[0]) + "\n")
        f.write("remove_urls=" + str(results[1]) + "\n")
        f.write("remove_stopwords=" + str(results[2]) + "\n")
        f.write("lemmatization=" + str(results[3]) + "\n")
        f.write("stemming=" + str(results[4]) + "\n")
        
    df = load_data(DATA_PATH)

    datastats.dataset_overview(df)
    datastats.missing_values(df)
    datastats.class_distribution(df)
    datastats.text_statistics(df, DATA_INFO_PATH)
    
    df["clean_post"] = df["post"].apply(
    lambda x: preprocessing.preprocess_text(
        x,
        lowercase=results[0],
        removeurls=results[1],
        removestopwords=results[2],
        lemmatize=results[3],
        stem=results[4]
        )
    )

    X = df["clean_post"]
    y = df["class_label"]
    
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=42
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
    )

    print("Train size:", len(X_train))
    print("Validation size:", len(X_val))
    print("Test size:", len(X_test))

    print("Training tokenizer...")

    MAX_WORDS = 10000
    MAX_LEN = 200

    tokenizer = Tokenizer(num_words=MAX_WORDS)
    tokenizer.fit_on_texts(X_train)

    with open(os.path.join(RUN_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq   = tokenizer.texts_to_sequences(X_val)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_val_pad   = pad_sequences(X_val_seq, maxlen=MAX_LEN)
    X_test_pad  = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    model = keras.Sequential([
        
        layers.Embedding(
            input_dim=MAX_WORDS,
            output_dim=128,
            input_length=MAX_LEN
        ),

        layers.Conv1D(128, 5, activation="relu"),

        layers.GlobalMaxPooling1D(),

        layers.Dense(64, activation="relu"),

        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])

    model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )

    model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=5,
    batch_size=32
    )

    model.save(os.path.join(RUN_DIR, "cnn_model.keras"))

    print("\nEvaluating model...")

    y_pred_prob = model.predict(X_test_pad)

    y_pred = (y_pred_prob > 0.5).astype("int32")

    print("Test accuracy:", accuracy_score(y_test, y_pred))
    print("Test precision:", precision_score(y_test, y_pred))
    print("Test recall:", recall_score(y_test, y_pred))
    print("Test F1:", f1_score(y_test, y_pred))