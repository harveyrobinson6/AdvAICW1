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
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    roc_curve,
    auc,
    precision_recall_curve
)

# -----------------------------
# CNN Hyperparameters
# -----------------------------
#'''
EMBEDDING_DIM = 128
FILTERS = 256
KERNEL_SIZE = 5

DENSE_UNITS = 128
DROPOUT_RATE = 0.5

EPOCHS = 10
BATCH_SIZE = 32
#'''
'''
EMBEDDING_DIM = 128
FILTERS = 128
KERNEL_SIZE = 5
DENSE_UNITS = 64
DROPOUT_RATE = 0.5

EPOCHS = 5
BATCH_SIZE = 32
'''

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

    CNN_PATH = os.path.join(RUN_DIR, "cnn_model.keras")
    CNN_CONFIG_PATH = os.path.join(RUN_DIR, "config.txt")
    NLP_CONFIG_PATH = os.path.join(RUN_DIR, "nlpconfig.txt")
    DATA_INFO_PATH = os.path.join(RUN_DIR, "datainfo.txt")
    THIS_RUN_TOKENIZER_PATH = os.path.join(RUN_DIR, "tokenizer.pkl")
    RESULTS_PATH = os.path.join(RUN_DIR, "metrics.txt")

    with open(NLP_CONFIG_PATH, "w") as f:
        f.write("lowercase=" + str(results[0]) + "\n")
        f.write("remove_urls=" + str(results[1]) + "\n")
        f.write("remove_stopwords=" + str(results[2]) + "\n")
        f.write("lemmatization=" + str(results[3]) + "\n")
        f.write("stemming=" + str(results[4]) + "\n")

    with open(CNN_CONFIG_PATH, "w") as f:
        f.write("CNN configuration\n")
        f.write(f"embedding_dim={EMBEDDING_DIM}\n")
        f.write(f"filters={FILTERS}\n")
        f.write(f"kernel_size={KERNEL_SIZE}\n")
        f.write(f"dense_units={DENSE_UNITS}\n")
        f.write(f"dropout={DROPOUT_RATE}\n")
        f.write(f"epochs={EPOCHS}\n")
        f.write(f"batch_size={BATCH_SIZE}\n")
        
    df = load_data(DATA_PATH)

    datastats.dataset_overview(df)
    datastats.missing_values(df)
    datastats.class_distribution(df)
    datastats.text_statistics(df, DATA_INFO_PATH)
    
    df["text"] = df["post"]

    if "news_headline" in df.columns:
        df["text"] = df["news_headline"].fillna("") + " " + df["post"]

    #"post"
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
    '''
    MAX_WORDS = 10000
    MAX_LEN = 200
    '''
    MAX_WORDS = 20000
    MAX_LEN = 300

    #, oov_token="<OOV>"
    #filters=''
    tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>")
    tokenizer.fit_on_texts(X_train)

    tokenizer.word_index = {
        k: v for k, v in tokenizer.word_index.items() if v <= MAX_WORDS
    }

    with open(os.path.join(RUN_DIR, "tokenizer.pkl"), "wb") as f:
        pickle.dump(tokenizer, f)

    early_stop = EarlyStopping(
    monitor="val_loss",
    patience=2,
    restore_best_weights=True
    )

    X_train_seq = tokenizer.texts_to_sequences(X_train)
    X_val_seq   = tokenizer.texts_to_sequences(X_val)
    X_test_seq  = tokenizer.texts_to_sequences(X_test)

    X_train_pad = pad_sequences(X_train_seq, maxlen=MAX_LEN)
    X_val_pad   = pad_sequences(X_val_seq, maxlen=MAX_LEN)
    X_test_pad  = pad_sequences(X_test_seq, maxlen=MAX_LEN)

    
    model = keras.Sequential([

        layers.Embedding(
        input_dim=MAX_WORDS,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_LEN
        ),

        layers.SpatialDropout1D(0.2),

        layers.Conv1D(FILTERS, KERNEL_SIZE, activation="relu"),

        layers.GlobalMaxPooling1D(),

        layers.Dense(DENSE_UNITS, activation="relu"),

        layers.Dropout(DROPOUT_RATE),

        layers.Dense(1, activation="sigmoid")
    ])
    '''deeper
    model = keras.Sequential([

        layers.Embedding(MAX_WORDS, EMBEDDING_DIM, input_length=MAX_LEN),

        layers.Conv1D(128, 5, activation="relu"),
        layers.MaxPooling1D(),

        layers.Conv1D(128, 5, activation="relu"),

        layers.GlobalMaxPooling1D(),

        layers.Dense(64, activation="relu"),
        layers.Dropout(0.5),

        layers.Dense(1, activation="sigmoid")
    ])
    '''


    model.compile(
    optimizer="adam",
    loss="binary_crossentropy",
    metrics=["accuracy"]
    )

    '''
    history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE
    )
    '''
    history = model.fit(
    X_train_pad,
    y_train,
    validation_data=(X_val_pad, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=[early_stop]
    )

    plt.figure()

    plt.plot(history.history["accuracy"])
    plt.plot(history.history["val_accuracy"])

    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend(["Train", "Validation"])

    plt.savefig(os.path.join(RUN_DIR, "training_accuracy.png"))
    plt.close()

    plt.figure()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])

    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend(["Train", "Validation"])

    plt.savefig(os.path.join(RUN_DIR, "training_loss.png"))
    plt.close()

    print("\nFinal Training Metrics")
    print("Train Accuracy:", history.history["accuracy"][-1])
    print("Validation Accuracy:", history.history["val_accuracy"][-1])
    print("Train Loss:", history.history["loss"][-1])
    print("Validation Loss:", history.history["val_loss"][-1])

    model.save(os.path.join(RUN_DIR, "cnn_model.keras"))

    print("\nEvaluating model...")

    # Probabilities
    y_probs = model.predict(X_test_pad).ravel()

    # Binary predictions
    y_pred = (y_probs > 0.5).astype("int32")

    # -----------------------------
    # Metrics
    # -----------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure()
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    plt.savefig(os.path.join(RUN_DIR, "confusion_matrix.png"))
    plt.close()

    # -----------------------------
    # ROC Curve
    # -----------------------------
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend()

    plt.savefig(os.path.join(RUN_DIR, "roc_curve.png"))
    plt.close()

    # -----------------------------
    # Precision Recall Curve
    # -----------------------------
    precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_probs)

    plt.figure()
    plt.plot(recall_curve, precision_curve)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")

    plt.savefig(os.path.join(RUN_DIR, "precision_recall_curve.png"))
    plt.close()

    # -----------------------------
    # Save metrics
    # -----------------------------
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"AUC: {roc_auc}\n")

    print("\nFinal Test Metrics")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("AUC:", roc_auc)