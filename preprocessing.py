import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer

def preprocess_text(text, lowercase=True, removeurls=True, removestopwords=True, lemmatize=False, stem=False):
    text = str(text)

    # lowercase
    if lowercase:
        text = text.lower()

    # remove URLs
    if removeurls:
        text = re.sub(r'http\S+|www\S+', '', text)

    # tokenize
    tokenizer = RegexpTokenizer(r'\w+')
    tokens = tokenizer.tokenize(text)

    # remove stopwords
    if removestopwords:
        custom_stopwords = {"read", "more", "click", "share"}
        stop_words = set(stopwords.words("english")).union(custom_stopwords)
        tokens = [w for w in tokens if w not in stop_words]

    # lemmatization
    if lemmatize:
        lemmatizer = WordNetLemmatizer()
        tokens = [lemmatizer.lemmatize(w) for w in tokens]

    # stemming
    if stem:
        stemmer = PorterStemmer()
        tokens = [stemmer.stem(w) for w in tokens]

    return " ".join(tokens)