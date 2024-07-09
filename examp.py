import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import re
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
            # Preprocess text
            preprocessed_text = preprocessor.preprocess_text(text)
            file_str.append(preprocessed_text)

    return ids, urls, file_str

def build_tfidf_matrix(file_str):
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(file_str)
    return tfidf_matrix, vectorizer

def cosine_similarity_query(query, tfidf_matrix, vectorizer):
    query_vec = vectorizer.transform([query])
    cosine_similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    return cosine_similarities

import numpy as np

def fetch_most_similar_documents(query, ids, urls, file_str, tfidf_matrix, vectorizer, num_documents=5):
    cosine_similarities = cosine_similarity_query(query, tfidf_matrix, vectorizer)
    # Sort the indices based on cosine similarity scores
    sorted_indices = np.argsort(cosine_similarities)[::-1][:num_documents]
    # Fetch the relevant document IDs and URLs
    relevant_doc_ids = [ids[i] for i in sorted_indices]
    relevant_doc_urls = [urls[i] for i in sorted_indices]
    return relevant_doc_ids, relevant_doc_urls


# Example usage:
if __name__ == "__main__":
    preprocessor = Preprocessor()
    filename = "sample.txt"
    query = preprocessor.preprocess_text("Indigenous people have patterns of illness very different from Western civilization; yet, they rapidly develop diseases once exposed to Western foods and lifestyles. Food and medicine were interwoven. All cultures used special or functional foods to prevent disease. Food could be used at different times either as food or medicine. Foods, cultivation, and cooking methods maximized community health and well-being. With methods passed down through generations, cooking processes were utilized that enhanced mineral and nutrient bioavailability. This article focuses on what researchers observed about the food traditions of indigenous people, their disease patterns, the use of specific foods, and the environmental factors that affect people who still eat traditional foods.")
    ids, urls, file_str = read_file(filename)
    tfidf_matrix, vectorizer = build_tfidf_matrix(file_str)
    relevant_doc_ids, relevant_doc_urls = fetch_most_similar_documents(query, ids, urls, file_str, tfidf_matrix, vectorizer, num_documents=5)
    print("5 Most Relevant Document IDs:", relevant_doc_ids)
    print("5 Most Relevant Document URLs:", relevant_doc_urls)
