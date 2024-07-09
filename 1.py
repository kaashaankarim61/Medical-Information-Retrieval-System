import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle

class Preprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, word):
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess_text(self, text):
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        tokens = word_tokenize(text)
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        lemmatized_tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens]
        stemmed_tokens = [self.stemmer.stem(token) for token in lemmatized_tokens]
        return " ".join(stemmed_tokens)

def read_file(filename, encoding='utf-8'):
    preprocessor = Preprocessor()
    ids = []
    urls = []
    file_str = []

    with open(filename, 'r', encoding=encoding) as file:
        for line in file:
            parts = line.strip().split('\t')
            doc_id, link, text = parts[0], parts[1], ' '.join(parts[2:])
            ids.append(doc_id)
            urls.append(link)
            preprocessed_text = preprocessor.preprocess_text(text)
            file_str.append(preprocessed_text)

    return ids, urls, file_str

def build_tfidf_matrix(file_str):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(file_str)
    return tfidf_matrix, vectorizer

if __name__ == "__main__":
    filename = "sample.txt"
    ids, urls, file_str = read_file(filename)
    tfidf_matrix, vectorizer = build_tfidf_matrix(file_str)
    
    # Save the TF-IDF matrix and vectorizer for later use
    with open("tfidf_matrix.pkl", "wb") as f:
        pickle.dump(tfidf_matrix, f)
    with open("vectorizer.pkl", "wb") as f:
        pickle.dump(vectorizer, f)
    # Save document IDs and URLs
    with open("document_ids.pkl", "wb") as f:
        pickle.dump(ids, f)
    with open("document_urls.pkl", "wb") as f:
        pickle.dump(urls, f)
