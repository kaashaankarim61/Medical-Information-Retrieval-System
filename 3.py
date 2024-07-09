import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import re
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('wordnet')
        self.stop_words = set(stopwords.words('english'))
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()

    def get_wordnet_pos(self, word):
        """Map POS tag to first character lemmatize() accepts"""
        tag = nltk.pos_tag([word])[0][1][0].upper()
        tag_dict = {"J": wordnet.ADJ,
                    "N": wordnet.NOUN,
                    "V": wordnet.VERB,
                    "R": wordnet.ADV}
        return tag_dict.get(tag, wordnet.NOUN)

    def preprocess_text(self, text):
        # Tokenize
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        tokens = word_tokenize(text)
        
        # Remove stop words
        tokens = [token for token in tokens if token.lower() not in self.stop_words]
        
        # Lemmatization
        lemmatized_tokens = [self.lemmatizer.lemmatize(token, self.get_wordnet_pos(token)) for token in tokens]
        
        # Stemming
        stemmed_tokens = [self.stemmer.stem(token) for token in lemmatized_tokens]
        
        # Return preprocessed text
        return " ".join(stemmed_tokens)

    def preprocess_string(self, input_string):
        return self.preprocess_text(input_string)

def cosine_similarity_query(query, tfidf_matrix, vectorizer):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return cosine_similarities

def fetch_most_similar_documents(query, ids, urls, tfidf_matrix, vectorizer, num_documents=5):
    cosine_similarities = cosine_similarity_query(query, tfidf_matrix, vectorizer)
    sorted_indices = np.argsort(cosine_similarities)[::-1][:num_documents]
    relevant_doc_ids = [ids[i] for i in sorted_indices]
    relevant_doc_urls = [urls[i] for i in sorted_indices]
    return relevant_doc_ids, relevant_doc_urls

def fetch_most_similar_documents_with_rocchio(query, ids, urls, tfidf_matrix, vectorizer, relevant_ids, nonrelevant_ids, alpha=1, beta=0.75, num_documents=5):
    # Convert query to TF-IDF representation
    query_vec = vectorizer.transform([query])
    query_vec = query_vec.toarray()  # Convert to numpy array

    # Convert tfidf_matrix to numpy array
    tfidf_matrix = tfidf_matrix.toarray()

    # Initialize Rocchio relevance feedback
    relevant_vecs = tfidf_matrix[np.where(np.isin(ids, relevant_ids))]
    nonrelevant_vecs = tfidf_matrix[np.where(np.isin(ids, nonrelevant_ids))]

    # Update query using Rocchio algorithm
    if relevant_vecs.shape[0] > 0:
        relevant_mean = np.mean(relevant_vecs, axis=0)
        query_vec += alpha * relevant_mean
    if nonrelevant_vecs.shape[0] > 0:
        nonrelevant_mean = np.mean(nonrelevant_vecs, axis=0)
        query_vec -= beta * nonrelevant_mean

    # Calculate cosine similarities
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()

    # Get most relevant documents
    sorted_indices = np.argsort(cosine_similarities)[::-1][:num_documents]
    relevant_doc_ids = [ids[i] for i in sorted_indices]
    relevant_doc_urls = [urls[i] for i in sorted_indices]

    return relevant_doc_ids, relevant_doc_urls

def update_relevant_documents():
    relevant = input("Enter relevant document IDs separated by spaces: ").split()
    return relevant

def update_nonrelevant_documents():
    nonrelevant = input("Enter non-relevant document IDs separated by spaces: ").split()
    return nonrelevant

if __name__ == "__main__":
    # Load preprocessed information
    with open("tfidf_matrix.pkl", "rb") as f:
        tfidf_matrix = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f:
        vectorizer = pickle.load(f)

    # Load document IDs and URLs
    with open("document_ids.pkl", "rb") as f:
        ids = pickle.load(f)
    with open("document_urls.pkl", "rb") as f:
        urls = pickle.load(f)

    preprocessor = Preprocessor()
    query = preprocessor.preprocess_text("Prostate")
    relevant_doc_ids, relevant_doc_urls = fetch_most_similar_documents(query, ids, urls, tfidf_matrix, vectorizer, num_documents=5)
    print("Initial 5 Most Relevant Document IDs:", relevant_doc_ids)
    print("Initial 5 Most Relevant Document URLs:", relevant_doc_urls)

    while True:
        relevant_ids = update_relevant_documents()
        nonrelevant_ids = update_nonrelevant_documents()

        relevant_doc_ids, relevant_doc_urls = fetch_most_similar_documents_with_rocchio(query, ids, urls, tfidf_matrix, vectorizer, relevant_ids, nonrelevant_ids, num_documents=5)
        print("Updated 5 Most Relevant Document IDs:", relevant_doc_ids)
        print("Updated 5 Most Relevant Document URLs:", relevant_doc_urls)

        choice = input("Continue relevance feedback? (yes/no): ").lower()
        if choice != 'yes':
            break
