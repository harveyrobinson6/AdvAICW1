import pandas as pd
import os
import io
import sys
import pickle
import datastats
import preprocessing
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_curve,
    auc,
    precision_recall_curve
)

def load_data(path):
    df = pd.read_csv(path)
    return df

if __name__ == "__main__":

    DATA_PATH = "social-media-release.csv"
    RUNS_DIR = "saves/mlp/runs"

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
    #CONFIG_PATH = os.path.join(RUN_DIR, "config.txt")
    NLP_CONFIG_PATH = os.path.join(RUN_DIR, "nlpconfig.txt")
    DATA_INFO_PATH = os.path.join(RUN_DIR, "datainfo.txt")
    THIS_RUN_TFIDF_PATH = os.path.join(RUN_DIR, "tfidf.pkl")
    RESULTS_PATH = os.path.join(RUN_DIR, "metrics.txt")
    MLP_CONFIG_PATH = os.path.join(RUN_DIR, "mlp_config.txt")
    
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
    
    print("Training TF-IDF vectoriser...")
    tfidf = TfidfVectorizer(
        #max_features=8000,
        max_features=15000,
        min_df=5,
        max_df=0.9,
        #ngram_range=(1, 2),
        ngram_range=(1, 3),
        sublinear_tf=True
    )
    tfidf.fit(X_train)

    with open(THIS_RUN_TFIDF_PATH, "wb") as f:
        pickle.dump(tfidf, f)

    # Vectorise (always fast)
    X_train_vec = tfidf.transform(X_train)
    X_val_vec   = tfidf.transform(X_val)
    X_test_vec  = tfidf.transform(X_test)

    print("TF-IDF train shape:", X_train_vec.shape)
    
    mlp = MLPClassifier(
            #hidden_layer_sizes=(128,),
            hidden_layer_sizes=(128,64),
            activation="relu",
            solver="adam",
            learning_rate_init=0.0005, #0.001
            max_iter=500, #200
            early_stopping=True,
            n_iter_no_change=10, #10
            random_state=42,
            alpha=0.0001
        )
    
    with open(MLP_CONFIG_PATH, "w") as f:
        for k, v in mlp.get_params().items():
            f.write(f"{k}: {v}\n")

    mlp.fit(X_train_vec, y_train)

    with open(MLP_PATH, "wb") as f:
        pickle.dump(mlp, f)
        
    # -----------------------------
    # TEST SET EVALUATION
    # -----------------------------

    # Predictions
    y_test_pred = mlp.predict(X_test_vec)
    y_probs = mlp.predict_proba(X_test_vec)[:, 1]

    # Metrics
    accuracy = accuracy_score(y_test, y_test_pred)
    precision = precision_score(y_test, y_test_pred)
    recall = recall_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred)

    # -----------------------------
    # Confusion Matrix
    # -----------------------------
    cm = confusion_matrix(y_test, y_test_pred)

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
    # Save metrics to file
    # -----------------------------
    with open(RESULTS_PATH, "w") as f:
        f.write(f"Accuracy: {accuracy}\n")
        f.write(f"Precision: {precision}\n")
        f.write(f"Recall: {recall}\n")
        f.write(f"F1: {f1}\n")
        f.write(f"AUC: {roc_auc}\n")

    # -----------------------------
    # Print results
    # -----------------------------
    print("\nFinal Test Metrics")
    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", f1)
    print("AUC:", roc_auc)